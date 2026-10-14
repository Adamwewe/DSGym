#!/usr/bin/env python3
"""
Orchestrate a DSGym DSPredict GPU smoke test on a Daytona sandbox.

This script drives a Daytona sandbox end-to-end: creates it (with a GPU),
clones DSGym into it, installs the manager + executor-kaggle Python
requirements, and runs the manager and executor workers as direct processes
inside the sandbox (no Docker-in-Docker). It runs `dsgym eval` against them
while streaming stdout/stderr live, and finally downloads
`evaluation_results/` (and `submissions/`) back to your laptop.

Phases (run with --phase, or `all` to run everything except teardown):
  create     Create the Daytona GPU sandbox and persist its id locally.
  preflight  Verify GPU and Linux mount-namespace support (used to isolate
             /submission and /data per executor). Fails fast if not available.
  bootstrap  Clone the repo into the sandbox and install Python dependencies
             for DSGym, manager, and executor-kaggle.
  launch     Generate a container_config.json for N executor-kaggle workers
             + 1 manager, start them as background processes, and wait for the
             manager /health endpoint before returning.
  run        Write KAGGLE_TOKEN to ~/.kaggle/access_token and run dsgym eval
             with the selected backend's credentials provisioned at create time,
             start `dsgym eval` in a background Daytona session, and stream its
             stdout/stderr live to this terminal until it finishes.
   fetch      Sync evaluation_results/ to the persistent volume, then tar and
              download the archive to --local-results-dir. Volume sync runs
              independently so results survive even when tar/download fails.
   auto-fetch Run fetch in a loop every --auto-fetch-interval seconds (default
              1200 = 20 min). Volume sync is attempted every cycle; laptop
              download may fail without losing results on the volume. Also runs
              automatically as a background thread during --phase run.
  teardown   Kill all manager/executor background processes inside the sandbox.
             By default the sandbox is only STOPPED afterwards (billing stops,
             filesystem preserved); add --delete-sandbox to also delete it
             (refused unless a successful `fetch` is on record, or --force).
  kill       Safety-net command: force-STOP a sandbox immediately (skips
             process cleanup) — this alone halts billing and leaves the
             filesystem/results untouched. Uses the state file's sandbox id
             by default, or --sandbox-id to target one directly (e.g. if the
             state file was lost). Add --delete-sandbox to also delete it
             (refused unless `fetch` succeeded for it, or --force).
  sweep      Another safety net: lists every sandbox this script has ever
             created (found by label, independent of any state file). Dry
             run by default; pass --yes to STOP all of them (billing stops,
             results preserved). Add --delete-sandbox --force on top of
             --yes only if you're certain results were already retrieved
             (or don't matter) — sweep can't verify `fetch` status per
             sandbox. Run this periodically, or whenever you're not sure if
             a sandbox from a past run is still alive and billing.
  all        create -> preflight -> bootstrap -> launch -> run -> fetch
             (does NOT stop/delete anything, so you can inspect the sandbox
             afterwards — remember to run `--phase kill` or
             `--phase sweep --yes` afterwards to stop billing; results are
             already safely downloaded to your laptop by `fetch` at that
             point, so stopping/deleting is safe).
   status     Read-only: lists every sandbox visible in your org/tenant
              (add --mine to filter to just this project's label) with its
              live state (creating/pending_build/building_snapshot/started/
              etc.), resources, and error_reason/build_info if present. Safe
              to run anytime, from a second terminal, even while `create` is
              still blocking elsewhere — useful for seeing whether a slow
              `create` is actually still provisioning/building a snapshot
              for a non-default image vs. stuck.
   exec       Run a one-off shell command inside the sandbox. Use --command
              (and optionally --exec-cwd). Useful for debugging, e.g.
              --command "curl -s http://localhost:5000/health".
   stop-run   Kill a running `dsgym eval` process inside the sandbox (does
              NOT stop the manager/executors or the sandbox). Use this when
              an eval is burning tokens on a broken setup.
   attach     Re-attach to a running eval session's live log stream. Finds
              the latest dsgym-eval-* session that's still running and
              re-streams its stdout/stderr. Use this when your laptop died
              or you Ctrl-C'd the orchestrator but the eval is still going.

State (sandbox id, repo dir, etc.) is persisted to --state-file (default:
daytona/.daytona_state.json) so phases can be re-run independently without
re-creating the sandbox each time.

Credentials (never committed, never baked into images):
  DAYTONA_API_KEY   Required. Auth for the Daytona SDK itself.
  OPENAI_API_KEY    Required for --backend litellm. Passed into the sandbox
                    as a secret env var at create time.
  AI_CODING_GATEWAY_API_KEY and AI_CODING_GATEWAY_PROJECT
                    Required for --backend gateway. Passed into the sandbox
                    as secret env vars at create time.
  KAGGLE_TOKEN      Required for DSPredict. Modern Kaggle access token
                    (starts with KGAT_...). Written to ~/.kaggle/access_token
                    inside the sandbox; never written to disk here.

Example:
  export DAYTONA_API_KEY=...
  export OPENAI_API_KEY=...
  python daytona/orchestrate.py --phase all --num-executors 4 --limit 5
"""

import argparse
import json
import os
import shlex
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

try:
    from daytona import (
        CreateSandboxFromImageParams,
        Daytona,
        GpuType,
        Image,
        Resources,
        SandboxState,
        SessionExecuteRequest,
        VolumeMount,
    )
except ImportError:
    print(
        "❌ The 'daytona' package is not installed in this environment.\n"
        "   Install it in its own venv so it never touches dsgym's pinned deps:\n"
        "     uv venv daytona/.venv --python 3.12\n"
        "     uv pip install --python daytona/.venv/bin/python -r daytona/requirements.txt\n"
        "     daytona/.venv/bin/python daytona/orchestrate.py ...\n",
        file=sys.stderr,
    )
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_FILE = SCRIPT_DIR / ".daytona_state.json"
MANAGER_PORT = 5000
EXECUTOR_BASE_PORT = 8432
EXECUTOR_START_PORT = 60000  # legacy host-port mapping kept for parity
# Upper bound for port-cleanup in _stop_services (8432 + 7 = 8439 for 8 executors).
MAX_EXECUTOR_WORKERS = 8
SERVICE_PORTS = [MANAGER_PORT] + [EXECUTOR_BASE_PORT + i for i in range(MAX_EXECUTOR_WORKERS)]
# Applied to every sandbox this script creates, so `sweep` can find (and
# kill) them later even without a local state file.
SANDBOX_LABELS = {"project": "dsgym", "purpose": "dspredict-gpu-smoketest"}
# Persistent volume — used for evaluation_results backup across sandbox deletion.
# Kaggle competition data is downloaded into the sandbox filesystem on each bootstrap
# (not cached on the volume; large volume copies have been unreliable).
KAGGLE_DATA_VOLUME_NAME = "dsgym-kaggle-data"
KAGGLE_DATA_VOLUME_MOUNT = "/mnt/kaggle-data"
# Maps --dataset values to (task JSON under repo root, data/data subdir used by dsgym eval).
_DSPREDICT_BOOTSTRAP_CONFIG: dict[str, tuple[str, str]] = {
    "dspredict-hard": ("data/task/dspredict/hard.json", "dspredict-hard"),
    "dspredict-easy": ("data/task/dspredict/easy.json", "dspredict-easy"),
    "dspredict-sampled": ("data/task/dspredict/dspredict_sampled.json", "dspredict-sampled"),
}
# Direct-process runtime layout
SANDBOX_DATA_DIR = "/data"
SANDBOX_SUBMISSION_DIR = "/submission"


def get_venv_python(repo_dir: str) -> str:
    """Absolute path to the uv-managed Python inside the repo's venv.

    `uv sync` creates `<repo>/.venv`; all manager/executor packages must be
    run with that interpreter, not the sandbox's system `python3`.
    """
    return f"{repo_dir.rstrip('/')}/.venv/bin/python3"


def get_manager_venv_python(repo_dir: str) -> str:
    """Absolute path to the Python inside the manager's isolated venv.

    The manager needs pinned FastAPI/Pydantic versions that conflict with the
    executor's dependency tree. Keeping it in a separate venv avoids the fight.
    """
    return f"{repo_dir.rstrip('/')}/.venv-manager/bin/python3"


def get_executor_venv_python(repo_dir: str) -> str:
    """Absolute path to the Python inside the executor's isolated venv.

    Same rationale as the manager venv: `uv sync` / `uv run` would otherwise
    wipe executor deps (fastapi, jupyter-client, etc.) from the repo `.venv`
    because they're not in the project's `uv.lock`.
    """
    return f"{repo_dir.rstrip('/')}/.venv-executor/bin/python3"


def get_repo_dir(sandbox, state: dict, args) -> str:
    """Repo checkout path inside the sandbox, cached in state after first computation.

    Computed from the sandbox's actual user home dir rather than hardcoded,
    since the home directory/user of a custom-image sandbox isn't guaranteed.
    """
    if "repo_dir" not in state:
        home = sandbox.get_user_home_dir()
        state["repo_dir"] = f"{home.rstrip('/')}/DSGym"
        save_state(args.state_file, state)
    return state["repo_dir"]


# --------------------------------------------------------------------------
# State persistence
# --------------------------------------------------------------------------

def load_state(state_file: Path) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {}


def save_state(state_file: Path, state: dict) -> None:
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_file)
    except Exception as e:
        print(f"⚠️  Failed to save state file {state_file}: {e}", file=sys.stderr)


def get_sandbox(daytona: Daytona, state: dict):
    sandbox_id = state.get("sandbox_id")
    if not sandbox_id:
        print("❌ No sandbox found in state file. Run --phase create first.", file=sys.stderr)
        sys.exit(1)
    try:
        return daytona.get(sandbox_id)
    except Exception as e:
        print(f"❌ Sandbox {sandbox_id} not found: {e}\n"
              "   It may have been killed/deleted. Run --phase create to provision a new one.",
              file=sys.stderr)
        sys.exit(1)


def run(sandbox, command: str, cwd: str = None, env: dict = None, timeout: int = None, check: bool = True):
    """Run a blocking command in the sandbox and print+return its output."""
    print(f"\n$ {command}")
    resp = sandbox.process.exec(command, cwd=cwd, env=env, timeout=timeout)
    out = resp.artifacts.stdout if resp.artifacts and resp.artifacts.stdout else resp.result
    if out:
        print(out)
    if check and resp.exit_code != 0:
        print(f"❌ Command failed (exit {resp.exit_code}): {command}", file=sys.stderr)
        sys.exit(1)
    return resp


# --------------------------------------------------------------------------
# Phase: create
# --------------------------------------------------------------------------

def _wait_for_sandbox_started(daytona: Daytona, sandbox_id: str, timeout: int = 1800):
    """Poll the sandbox ourselves until it's STARTED (or a terminal failure
    state), instead of relying solely on the SDK's own internal wait —
    that internal wait can raise on a transient 502/network hiccup even
    though the sandbox is still being built/started fine server-side."""
    terminal_failures = {SandboxState.ERROR, SandboxState.BUILD_FAILED, SandboxState.DESTROYED}
    last_state = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sandbox = daytona.get(sandbox_id)
        except Exception as e:
            print(f"⚠️  Transient error polling sandbox state ({e}); retrying...")
            time.sleep(10)
            continue
        if sandbox.state != last_state:
            print(f"   ... state={sandbox.state}")
            last_state = sandbox.state
        if sandbox.state == SandboxState.STARTED:
            return sandbox
        if sandbox.state in terminal_failures:
            print(f"❌ Sandbox {sandbox_id} entered {sandbox.state}"
                  f"{f' ({sandbox.error_reason})' if getattr(sandbox, 'error_reason', None) else ''}.",
                  file=sys.stderr)
            sys.exit(1)
        time.sleep(10)
    print(f"❌ Timed out after {timeout}s waiting for sandbox {sandbox_id} to start "
          f"(last state={last_state}). It may still be building — check with "
          f"`--phase status --mine` and re-run `--phase create --adopt-sandbox-id {sandbox_id}` "
          f"once it's further along, or just wait and check status again.", file=sys.stderr)
    sys.exit(1)


def _find_recent_orphaned_sandbox(daytona: Daytona, max_age_seconds: int = 900):
    """Best-effort recovery for when daytona.create() raises (e.g. a 502 while
    the SDK was polling) but the sandbox was actually created server-side.
    Looks for our-labeled sandboxes created very recently and still alive."""
    from daytona import ListSandboxesQuery

    candidates = list(daytona.list(ListSandboxesQuery(labels=SANDBOX_LABELS)))
    now = time.time()
    alive = [
        sb for sb in candidates
        if sb.state not in (SandboxState.DESTROYED, SandboxState.DESTROYING)
    ]
    # created_at is an ISO8601 string; sort newest first and sanity-check age.
    def _age_ok(sb) -> bool:
        try:
            from datetime import datetime, timezone
            created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
            return (now - created.timestamp()) <= max_age_seconds
        except Exception:
            return True  # if we can't parse it, don't filter it out

    recent = sorted(
        (sb for sb in alive if _age_ok(sb)),
        key=lambda sb: sb.created_at or "",
        reverse=True,
    )
    return recent[0] if recent else None


def phase_create(daytona: Daytona, state: dict, args) -> dict:
    if state.get("sandbox_id"):
        print(f"ℹ️  Sandbox already exists in state ({state['sandbox_id']}), skipping create. "
              f"Delete the state file to force a new one.")
        return state

    if args.adopt_sandbox_id:
        print(f"🔗 Adopting existing sandbox {args.adopt_sandbox_id} instead of creating a new one...")
        sandbox = daytona.get(args.adopt_sandbox_id)
        state["sandbox_id"] = sandbox.id
        state["created_at"] = time.time()
        save_state(args.state_file, state)
        sandbox = _wait_for_sandbox_started(daytona, sandbox.id, timeout=args.create_timeout)
        print(f"✅ Sandbox ready: id={sandbox.id} (state={sandbox.state})")
        return state

    gpu_types = None
    if args.gpu_type:
        gpu_types = [getattr(GpuType, t.strip().upper()) for t in args.gpu_type.split(",")]
    if args.backend == "gateway":
        gateway_key = os.environ.get("AI_CODING_GATEWAY_API_KEY")
        gateway_project = os.environ.get("AI_CODING_GATEWAY_PROJECT")
        missing_gateway_vars = [
            name for name, value in (
                ("AI_CODING_GATEWAY_API_KEY", gateway_key),
                ("AI_CODING_GATEWAY_PROJECT", gateway_project),
            ) if not value
        ]
        if missing_gateway_vars:
            print(
                "❌ Gateway backend requires "
                f"{', '.join(missing_gateway_vars)} in the local environment. "
                "Export them before --phase create; the sandbox receives them "
                "as environment variables.",
                file=sys.stderr,
            )
            sys.exit(1)
        inference_env = {
            "AI_CODING_GATEWAY_API_KEY": gateway_key,
            "AI_CODING_GATEWAY_PROJECT": gateway_project,
        }
    elif args.backend == "litellm":
        openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("❌ OPENAI_API_KEY not set locally and --openai-api-key not passed. "
                  "The sandbox needs it at creation time so `dsgym eval` can read it later.", file=sys.stderr)
            sys.exit(1)
        inference_env = {"OPENAI_API_KEY": openai_key}
    else:
        inference_env = {}

    print(f"🚀 Creating Daytona GPU sandbox (cpu={args.cpu}, memory={args.memory}GB, "
          f"disk={args.disk}GB, gpu={args.gpu_count}, gpu_type={gpu_types or 'any available'})...")

    # Create or get the persistent Kaggle data volume.
    print(f"💾 Ensuring volume '{KAGGLE_DATA_VOLUME_NAME}' exists...")
    volume = daytona.volume.get(KAGGLE_DATA_VOLUME_NAME, create=True)
    # Wait for volume to be ready (max 2 min)
    for _ in range(120):
        if volume.state == "ready":
            break
        time.sleep(1)
        volume = daytona.volume.get(KAGGLE_DATA_VOLUME_NAME)
    if volume.state != "ready":
        print(f"⚠️  Volume '{KAGGLE_DATA_VOLUME_NAME}' not ready after 2 min (state={volume.state}). "
              "Proceeding without volume mount — Kaggle data will need to be re-downloaded.")
        volume_mounts = []
    else:
        print(f"   Volume ready (id={volume.id}). Will mount at {KAGGLE_DATA_VOLUME_MOUNT}.")
        volume_mounts = [VolumeMount(
            volume_id=volume.id,
            mount_path=KAGGLE_DATA_VOLUME_MOUNT,
        )]

    image = (
        Image.base("ubuntu:22.04")
        .run_commands(
            "apt-get update",
            "apt-get install -y python3 python3-pip python3-dev python3-venv "
            "build-essential curl git ca-certificates gnupg sudo lsb-release wget",
            "apt-get clean",
            "rm -rf /var/lib/apt/lists/*",
        )
    )
    params = CreateSandboxFromImageParams(
        image=image,
        resources=Resources(
            cpu=args.cpu,
            memory=args.memory,
            disk=args.disk,
            gpu=args.gpu_count,
            gpu_type=gpu_types,
        ),
        env_vars=inference_env,
        auto_delete_interval=0,  # GPU sandboxes must be ephemeral (Daytona requirement)
        auto_stop_interval=999999,  # ~694 days — effectively "never auto-stop"
        labels=SANDBOX_LABELS,
        volumes=volume_mounts,
    )
    try:
        sandbox = daytona.create(params)
    except Exception as e:
        print(f"⚠️  daytona.create() raised an error: {e}")
        print(
            "   This commonly happens when the SDK's own internal wait-for-start hits a "
            "transient gateway error (e.g. 502) while Daytona is still building/starting the "
            "sandbox server-side — the sandbox itself may well still be alive. Attempting to "
            "find and adopt it automatically..."
        )
        recovered = _find_recent_orphaned_sandbox(daytona)
        if recovered is None:
            print(
                "❌ Could not find a recently-created matching sandbox to adopt. Run "
                "`--phase status --mine` to check manually — if you find one, re-run with "
                "`--phase create --adopt-sandbox-id <id>` to adopt it instead of creating a "
                "new one (avoids paying the snapshot-build cost twice).",
                file=sys.stderr,
            )
            raise
        print(f"🔗 Found and adopting {recovered.id} (state={recovered.state}, "
              f"created_at={recovered.created_at}).")
        state["sandbox_id"] = recovered.id
        state["created_at"] = time.time()
        save_state(args.state_file, state)
        sandbox = _wait_for_sandbox_started(daytona, recovered.id, timeout=args.create_timeout)

    print(f"✅ Sandbox created: id={sandbox.id}")

    state["sandbox_id"] = sandbox.id
    state["created_at"] = time.time()
    save_state(args.state_file, state)
    return state


# --------------------------------------------------------------------------
# Direct-process helpers
# --------------------------------------------------------------------------

def _ensure_system_packages(sandbox) -> None:
    """Install the minimal system packages needed for building Python packages and extracting data."""
    print("🔧 Installing system packages for Python ML builds...")
    run(sandbox, (
        "apt-get update && apt-get install -y "
        "build-essential libgomp1 curl ca-certificates gnupg git unzip lsof "
        "libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx"
    ), timeout=300)


def _install_manager_deps(sandbox, repo_dir: str) -> None:
    """Install the manager's Python dependencies into a dedicated venv.

    The manager needs an older FastAPI/Pydantic/httpx stack than the rest of
    DSGym/executor-kaggle. A separate venv avoids resolver fights and ensures the
    manager always starts with the exact pinned versions.
    """
    print("🔧 Creating manager venv and installing manager dependencies...")
    venv_dir = f"{repo_dir.rstrip('/')}/.venv-manager"
    venv_python = get_manager_venv_python(repo_dir)
    run(sandbox, (
        f"$HOME/.local/bin/uv venv --python 3.12 --clear {shlex.quote(venv_dir)}"
    ), cwd=repo_dir, timeout=120)
    run(sandbox, (
        f"$HOME/.local/bin/uv pip install --python {shlex.quote(venv_python)} -r "
        f"{shlex.quote(repo_dir)}/executors/manager/requirements.txt"
    ), cwd=repo_dir, timeout=300)
    fa_version = sandbox.process.exec(
        f"{shlex.quote(venv_python)} -c 'import fastapi; print(fastapi.__version__)'",
        cwd=repo_dir,
    )
    fa_version_str = fa_version.result.strip() if fa_version.result else "unknown"
    print(f"   Manager venv FastAPI version: {fa_version_str}")


def _install_executor_deps(sandbox, repo_dir: str) -> None:
    """Install the executor-kaggle Python dependencies into a dedicated venv.

    The executor needs fastapi/uvicorn/jupyter-client/ipykernel plus all the ML
    libraries. Installing into the repo `.venv` doesn't work because `uv sync`
    / `uv run` wipes anything not in `uv.lock`. A separate `.venv-executor`
    keeps them safe.
    """
    print("🔧 Creating executor venv and installing executor-kaggle dependencies...")
    venv_dir = f"{repo_dir.rstrip('/')}/.venv-executor"
    venv_python = get_executor_venv_python(repo_dir)
    run(sandbox, (
        f"$HOME/.local/bin/uv venv --python 3.12 --clear {shlex.quote(venv_dir)}"
    ), cwd=repo_dir, timeout=120)
    run(sandbox, (
        f"$HOME/.local/bin/uv pip install --python {shlex.quote(venv_python)} -r "
        f"{shlex.quote(repo_dir)}/executors/container_images/kaggle_image/requirements.kaggle.txt"
    ), cwd=repo_dir, timeout=600)

    print("🔧 Upgrading torch to CUDA 12.6 wheels in the executor venv...")
    run(sandbox, (
        f"$HOME/.local/bin/uv pip install --python {shlex.quote(venv_python)} --force-reinstall --index-url "
        "https://download.pytorch.org/whl/cu126 torch"
    ), cwd=repo_dir, timeout=600)

    print("🔎 Verifying installed PyTorch can see the GPU...")
    gpu_check = sandbox.process.exec(
        f"{shlex.quote(venv_python)} -c \"import torch; print('torch version:', torch.__version__); "
        f"print('CUDA available:', torch.cuda.is_available()); "
        f"print('CUDA devices:', torch.cuda.device_count())\"",
        cwd=repo_dir,
    )
    print(gpu_check.result)
    if gpu_check.exit_code != 0 or "CUDA available: False" in gpu_check.result:
        print(
            "❌ PyTorch installed but CUDA is not available.\n"
            "   Common causes:\n"
            "   - torch was installed from the CPU-only wheel instead of the CUDA wheel\n"
            "   - the CUDA runtime libraries are not compatible with the driver in the sandbox\n"
            "   Try: uv pip list | grep torch; and check nvidia-smi shows a driver version.",
            file=sys.stderr,
        )
        sys.exit(1)


def _wait_for_manager(sandbox, timeout: int = 120):
    print("⏳ Waiting for manager /health to report ready...")
    for attempt in range(timeout):
        resp = sandbox.process.exec(f"curl -sf http://localhost:{MANAGER_PORT}/health")
        if resp.exit_code == 0:
            print(f"✅ Manager healthy after {attempt}s.")
            return
        time.sleep(1)
    print("❌ Manager never became healthy. Check /tmp/dsgym_manager.log in the sandbox.", file=sys.stderr)
    sys.exit(1)


def _wait_for_executor(sandbox, port: int, timeout: int = 180):
    for attempt in range(timeout):
        resp = sandbox.process.exec(f"curl -sf http://localhost:{port}/ready")
        if resp.exit_code == 0:
            return True
        time.sleep(1)
    return False


def _pids_on_port_shell() -> str:
    """Shell snippet that prints PIDs listening on $port (must be set by caller)."""
    return (
        "if command -v lsof >/dev/null 2>&1; then "
        "  lsof -ti :$port 2>/dev/null || true; "
        "elif command -v ss >/dev/null 2>&1; then "
        "  ss -H -tlnp sport = :$port 2>/dev/null "
        "    | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true; "
        "elif command -v fuser >/dev/null 2>&1; then "
        "  fuser -n tcp $port 2>/dev/null || true; "
        "fi"
    )


_pids_on_port = _pids_on_port_shell()


def _ports_in_use(sandbox, ports: list[int]) -> list[int]:
    """Return which of *ports* still have a listener inside the sandbox."""
    port_checks = " ".join(str(p) for p in ports)
    script = (
        f"busy=''; for port in {port_checks}; do "
        f"  pids=$({_pids_on_port}); "
        "  [ -n \"$pids\" ] && busy=\"$busy $port\"; "
        "done; echo $busy"
    )
    resp = sandbox.process.exec(script)
    raw = (resp.result or "").strip()
    if not raw:
        return []
    return [int(p) for p in raw.split()]


def _stop_services(sandbox) -> None:
    """Kill any manager/executor processes started by this orchestrator."""
    print("🧹 Stopping manager + executor processes...")
    port_list = " ".join(str(p) for p in SERVICE_PORTS)
    sandbox.process.exec(
        # Kill using saved pid files, then any child processes.
        "for f in /tmp/dsgym_manager.pid /tmp/dsgym_executor_*.pid; do "
        "  [ -f \"$f\" ] || continue; "
        "  pid=$(cat \"$f\" 2>/dev/null); "
        "  [ -n \"$pid\" ] && kill -9 \"$pid\" 2>/dev/null || true; "
        "  [ -n \"$pid\" ] && pkill -9 -P \"$pid\" 2>/dev/null || true; "
        "done; "
        # Kill by known process signatures (survives exec/unshare renames).
        "pkill -9 -f 'unshare --mount bash -c' || true; "
        "pkill -9 -f '/\\.venv-executor/bin/python3' || true; "
        "pkill -9 -f '/\\.venv-executor/bin/python main\\.py' || true; "
        "pkill -9 -f '/\\.venv-manager/bin/python3' || true; "
        "pkill -9 -f '/\\.venv-manager/bin/python main\\.py' || true; "
        "pkill -9 -f 'ipykernel_launcher' || true; "
        "pkill -9 -f 'executors/manager/main.py' || true; "
        "pkill -9 -f 'container_images/kaggle_image/main.py' || true; "
        "pkill -9 -f '/tmp/dsgym_executor_' || true; "
        # Kill anything still listening on our ports.
        f"for port in {port_list}; do "
        f"  for pid in $({_pids_on_port}); do "
        "    [ -n \"$pid\" ] && kill -9 \"$pid\" 2>/dev/null || true; "
        "  done; "
        "done; "
        "rm -f /tmp/dsgym_executor_*.pid /tmp/dsgym_manager.pid; "
        "sleep 2"
    )
    busy = _ports_in_use(sandbox, SERVICE_PORTS)
    if busy:
        print(
            f"⚠️  Ports still in use after cleanup: {busy}. Retrying force-kill...",
            file=sys.stderr,
        )
        sandbox.process.exec(
            f"for port in {' '.join(str(p) for p in busy)}; do "
            f"  for pid in $({_pids_on_port}); do "
            "    [ -n \"$pid\" ] && kill -9 \"$pid\" 2>/dev/null || true; "
            "  done; "
            "done; sleep 1"
        )
        busy = _ports_in_use(sandbox, SERVICE_PORTS)
        if busy:
            print(
                f"❌ Could not free ports {busy}. "
                "Install `lsof` in the sandbox (`--phase bootstrap` on a fresh sandbox) "
                "or stop the sandbox and re-create.",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------
# Phase: preflight  (GPU + mount namespace support)
# --------------------------------------------------------------------------

def phase_preflight(sandbox, state: dict, args) -> dict:
    print("🔎 Verifying GPU visibility inside the sandbox...")
    run(sandbox, "nvidia-smi", check=True)

    print("🔎 Verifying Linux mount namespace support (needed for per-executor /submission)...")
    run(sandbox, "mkdir -p /tmp/dsgym_ns_test/src /tmp/dsgym_ns_test/target")
    ns_test = sandbox.process.exec(
        "unshare --mount bash -c "
        "'mount --bind /tmp/dsgym_ns_test/src /tmp/dsgym_ns_test/target && "
        "touch /tmp/dsgym_ns_test/target/hello.txt' && "
        "test -f /tmp/dsgym_ns_test/src/hello.txt"
    )
    if ns_test.exit_code != 0:
        print(
            "❌ Cannot create a mount namespace or bind-mount inside it. "
            "This is required to isolate /submission and /data per executor.\n"
            "   Output: " + ns_test.result,
            file=sys.stderr,
        )
        sys.exit(1)
    print("   ✅ Mount namespace + bind mount works.")
    run(sandbox, "rm -rf /tmp/dsgym_ns_test")

    print("✅ Preflight passed: GPU + mount namespaces both work.")
    state["preflight_ok"] = True
    save_state(args.state_file, state)
    return state


# --------------------------------------------------------------------------
# Phase: bootstrap  (clone + install deps)
# --------------------------------------------------------------------------

def _validate_dspredict_bootstrap_data(sandbox, repo_dir: str, dataset: str) -> None:
    """Ensure every required competition has real files under data/data/{split}/.

    Fails bootstrap (sys.exit) if any competition dir is missing, is a
    .download_failed stub, or has no data files. Matches the paths used by
    dsgym.datasets.loaders.dspredict (e.g. data/data/dspredict-hard/{name}/).

    Uses per-competition checks with short timeouts so a slow volume mount
    surfaces as a specific competition (or timeout) rather than hanging bootstrap.
    """
    if dataset not in _DSPREDICT_BOOTSTRAP_CONFIG:
        return

    task_json_rel, data_subdir = _DSPREDICT_BOOTSTRAP_CONFIG[dataset]
    validation_script = f"""\
import json
import os
import subprocess
import sys

repo_dir = {repo_dir!r}
dataset = {dataset!r}
task_json_rel = {task_json_rel!r}
data_subdir = {data_subdir!r}
data_root = os.path.join(repo_dir, "data", "data", data_subdir)
FIND_TIMEOUT_SEC = 30


def competition_dir_name(entry: dict) -> str:
    path = entry.get("docker_challenge_path") or entry.get("challenge_name") or ""
    return path.rstrip("/").split("/")[-1]


def has_real_data(comp_dir: str) -> bool | None:
    \"\"\"Return True/False, or None if the check timed out.\"\"\"
    if not os.path.isdir(comp_dir):
        return False
    try:
        proc = subprocess.run(
            [
                "find",
                comp_dir,
                "-maxdepth",
                "2",
                "-type",
                "f",
                "!",
                "-name",
                ".gitkeep",
                "!",
                "-name",
                ".download_failed",
                "-print",
                "-quit",
            ],
            capture_output=True,
            text=True,
            timeout=FIND_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None
    return proc.returncode == 0 and bool(proc.stdout.strip())


def load_competition_names() -> list[str]:
    task_json = os.path.join(repo_dir, task_json_rel)
    if os.path.isfile(task_json):
        with open(task_json, encoding="utf-8") as f:
            items = json.load(f)
        return [competition_dir_name(d) for d in items]
    if dataset == "dspredict-sampled" and os.path.isdir(data_root):
        return sorted(
            name
            for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name))
        )
    print(f"❌ Task list not found: {{task_json}}", file=sys.stderr)
    sys.exit(1)


if not os.path.isdir(data_root):
    print(f"❌ Competition data root missing: {{data_root}}", file=sys.stderr)
    sys.exit(1)

competition_names = load_competition_names()
missing: list[str] = []
stub_only: list[str] = []
empty: list[str] = []
timed_out: list[str] = []

for i, name in enumerate(competition_names, 1):
    comp_dir = os.path.join(data_root, name)
    print(f"   [{{i}}/{{len(competition_names)}}] {{name}}...", flush=True)
    if not os.path.isdir(comp_dir):
        missing.append(name)
        continue
    has_data = has_real_data(comp_dir)
    if has_data is None:
        timed_out.append(name)
    elif os.path.isfile(os.path.join(comp_dir, ".download_failed")) and not has_data:
        stub_only.append(name)
    elif not has_data:
        empty.append(name)

problems = missing + stub_only + empty + timed_out
if not problems:
    print(f"✅ Competition data validated: {{len(competition_names)}} competitions under {{data_root}}")
    sys.exit(0)

print(f"❌ Bootstrap data validation failed for {{dataset}} (expected under {{data_root}}):", file=sys.stderr)
if missing:
    print(f"   Missing dirs ({{len(missing)}}): {{', '.join(missing[:10])}}{{'...' if len(missing) > 10 else ''}}", file=sys.stderr)
if stub_only:
    print(f"   Stub only (.download_failed, {{len(stub_only)}}): {{', '.join(stub_only[:10])}}{{'...' if len(stub_only) > 10 else ''}}", file=sys.stderr)
if empty:
    print(f"   Empty dirs ({{len(empty)}}): {{', '.join(empty[:10])}}{{'...' if len(empty) > 10 else ''}}", file=sys.stderr)
if timed_out:
    print(f"   Check timed out ({{len(timed_out)}}): {{', '.join(timed_out[:10])}}{{'...' if len(timed_out) > 10 else ''}}", file=sys.stderr)
    print(
        "   Slow volume I/O or very large competition dirs can cause this — "
        "not necessarily missing data.",
        file=sys.stderr,
    )
sys.exit(1)
"""
    sandbox.fs.upload_file(validation_script.encode("utf-8"), "/tmp/dsgym_validate_data.py")
    print(f"🔎 Validating competition data for {dataset} under {repo_dir}/data/data/...")
    try:
        resp = sandbox.process.exec(
            "python3 /tmp/dsgym_validate_data.py",
            cwd=repo_dir,
            timeout=3600,
        )
    except Exception as e:
        err = str(e)
        if "timeout" in err.lower():
            print(
                "❌ Data validation timed out before completing all checks.\n"
                "   This is an infrastructure/performance issue, not proof that data is missing.\n"
                "   Re-run bootstrap to stream per-competition progress, or inspect manually:\n"
                f"     orchestrate.py --phase exec --command 'ls {repo_dir}/data/data/{data_subdir} | head'",
                file=sys.stderr,
            )
        raise
    if resp.result:
        print(resp.result.rstrip())
    if resp.exit_code != 0:
        sys.exit(1)


def phase_bootstrap(sandbox, state: dict, args) -> dict:
    repo_dir = get_repo_dir(sandbox, state, args)
    print(f"📥 Cloning {args.repo_url} (branch: {args.branch}) into {repo_dir}...")
    check = sandbox.process.exec(f"test -d {repo_dir}/.git")
    if check.exit_code != 0:
        sandbox.git.clone(
            url=args.repo_url,
            path=repo_dir,
            branch=args.branch,
            username=args.github_user,
            password=args.github_token,
        )
    else:
        print("ℹ️  Repo already cloned, pulling latest instead.")
        run(sandbox, f"git -C {repo_dir} pull")

    print("🐍 Installing uv and syncing DSGym's Python dependencies (needed for the `dsgym` CLI)...")
    check_uv = sandbox.process.exec("command -v uv")
    if check_uv.exit_code != 0:
        run(sandbox, "curl -LsSf https://astral.sh/uv/install.sh | sh")
    run(sandbox, "$HOME/.local/bin/uv sync", cwd=repo_dir, timeout=900)

    # Fresh output dir for this sandbox run. Previous results live on the volume
    # (synced by fetch/auto-fetch) and are not needed before --phase run.
    run(sandbox, f"mkdir -p {repo_dir}/evaluation_results", check=False)

    _ensure_system_packages(sandbox)
    _install_manager_deps(sandbox, repo_dir)
    _install_executor_deps(sandbox, repo_dir)

    print("📦 Upgrading Kaggle CLI/sdk to support modern access-token auth...")
    venv_python = get_venv_python(repo_dir)
    run(sandbox, f"$HOME/.local/bin/uv pip install --python {shlex.quote(venv_python)} 'kaggle>=2.2.2'", cwd=repo_dir, timeout=300)

    kaggle_token = os.environ.get("KAGGLE_TOKEN")
    if not kaggle_token:
        print("❌ KAGGLE_TOKEN not set. Export your Kaggle access token before running:\n"
              "   export KAGGLE_TOKEN=KGAT_...", file=sys.stderr)
        sys.exit(1)
    print("🔑 Writing KAGGLE_TOKEN to ~/.kaggle/access_token inside the sandbox...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(kaggle_token)
        tmp_path = tmp.name
    try:
        sandbox.fs.upload_file(tmp_path, "/tmp/kaggle_access_token")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    run(sandbox, "mkdir -p ~/.kaggle && mv /tmp/kaggle_access_token ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token")

    # ── Kaggle data: always download into sandbox local disk ─────────────────
    # Do not symlink from the persistent volume — partial/stub caches and large
    # volume copies have caused missing competition data. The volume is kept for
    # evaluation_results backup only (see fetch / auto-fetch).
    data_base = f"{repo_dir}/data/data"
    run(sandbox, f"mkdir -p {data_base}")

    download_script = (
        "download_dspredict_data.sh"
        if args.dataset == "dspredict-sampled"
        else "download_dspredict_data_full.sh"
    )
    download_dir = (
        f"{data_base}/dspredict-sampled"
        if args.dataset == "dspredict-sampled"
        else f"{data_base}/dspredict"
    )
    run(
        sandbox,
        f"if [ -L {shlex.quote(download_dir)} ]; then "
        f"rm {shlex.quote(download_dir)}; fi; "
        f"mkdir -p {shlex.quote(download_dir)}",
    )
    print(f"📥 Downloading Kaggle competition data via {download_script} (10–30 min)...")
    run(sandbox, f"PATH=$HOME/.local/bin:$PATH $HOME/.local/bin/uv run kaggle --version", cwd=repo_dir)
    run(sandbox, (
        f"PATH=$HOME/.local/bin:$PATH "
        f"bash {download_script}"
    ), cwd=repo_dir, timeout=3600)

    # ── Post-download data setup ────────────────────────────────────────────
    # Each downloaded collection is the source for the split directories.
    source_dir = download_dir
    split_dirs = (
        ("dspredict", "dspredict-hard", "dspredict-easy")
        if args.dataset == "dspredict-sampled"
        else ("dspredict-sampled", "dspredict-hard", "dspredict-easy")
    )
    for split in split_dirs:
        split_path = f"{data_base}/{split}"
        run(
            sandbox,
            f"rm -rf {shlex.quote(split_path)} && "
            f"ln -s {shlex.quote(source_dir)} {shlex.quote(split_path)}",
            check=False,
        )

    _validate_dspredict_bootstrap_data(sandbox, repo_dir, args.dataset)

    state["bootstrap_ok"] = True
    save_state(args.state_file, state)
    return state


# --------------------------------------------------------------------------
# Phase: launch  (start manager + executors as direct processes)
# --------------------------------------------------------------------------

def phase_launch(sandbox, state: dict, args) -> dict:
    repo_dir = get_repo_dir(sandbox, state, args)
    n = args.num_executors

    # Clean up any stale processes from previous launches.
    _stop_services(sandbox)
    busy = _ports_in_use(sandbox, SERVICE_PORTS[: 1 + n])
    if busy:
        print(f"❌ Cannot launch: port(s) {busy} still in use after cleanup.", file=sys.stderr)
        sys.exit(1)

    # Ensure the data and submission directories exist.
    run(sandbox, f"mkdir -p {repo_dir}/data/data {repo_dir}/submissions")
    run(sandbox, f"mkdir -p {SANDBOX_DATA_DIR} {SANDBOX_SUBMISSION_DIR}")
    run(sandbox, f"chmod 777 {SANDBOX_SUBMISSION_DIR}")

    # Generate container_config.json for the manager.
    config_path = f"{repo_dir}/daytona_container_config.json"
    print(f"🧩 Generating {config_path} for {n} executor-kaggle worker(s)...")
    config = {
        str(i): {
            "url": f"http://localhost:{EXECUTOR_BASE_PORT + i}",
            "type": "executor-kaggle",
        }
        for i in range(n)
    }
    config_json = json.dumps(config, indent=2)
    run(sandbox, f"cat > {shlex.quote(config_path)} <<'EOF'\n{config_json}\nEOF")

    # Create per-executor submission directories.
    for i in range(n):
        run(sandbox, f"mkdir -p {repo_dir}/submissions/container_{i:03d} && chmod 777 {repo_dir}/submissions/container_{i:03d}")

    manager_venv_python = get_manager_venv_python(repo_dir)
    fa_version = sandbox.process.exec(
        f"{shlex.quote(manager_venv_python)} -c 'import fastapi; print(fastapi.__version__)'",
        cwd=repo_dir,
    )
    print(f"   Manager venv FastAPI version before launch: {fa_version.result.strip() if fa_version.result else 'unknown'}")

    print(f"🚀 Starting manager on port {MANAGER_PORT}...")
    manager_cmd = (
        f"cd {shlex.quote(repo_dir)}/executors/manager && "
        f"PATH={shlex.quote(os.path.dirname(manager_venv_python))}:$PATH "
        f"PORT={MANAGER_PORT} "
        f"CONTAINER_CONFIG_PATH={shlex.quote(config_path)} "
        f"ALLOCATION_TIMEOUT={args.allocation_timeout} "
        f"nohup {shlex.quote(manager_venv_python)} main.py > /tmp/dsgym_manager.log 2>&1 & "
        f"echo $! > /tmp/dsgym_manager.pid"
    )
    run(sandbox, manager_cmd, check=False)
    run(sandbox, "sleep 2")

    _wait_for_manager(sandbox)

    print(f"🚀 Starting {n} executor-kaggle worker(s) with mount-namespace isolation...")
    executor_venv_python = get_executor_venv_python(repo_dir)
    for i in range(n):
        port = EXECUTOR_BASE_PORT + i
        host_port = EXECUTOR_START_PORT + i
        submission_dir = f"{repo_dir}/submissions/container_{i:03d}"
        log_path = f"/tmp/dsgym_executor_{i:03d}.log"
        pid_path = f"/tmp/dsgym_executor_{i:03d}.pid"
        executor_dir = f"{repo_dir}/executors/container_images/kaggle_image"
        wrapper_path = f"/tmp/dsgym_executor_{i:03d}.sh"

        # Each executor runs in its own mount namespace so that /data and
        # /submission are isolated per worker (emulating the Docker volume mounts).
        wrapper_script = (
            "#!/bin/bash\n"
            f"export PORT={port}\n"
            f"export CONTAINER_ID={i}\n"
            f"export HOST_PORT={host_port}\n"
            f"export EXECUTION_TIMEOUT={args.execution_timeout}\n"
            f"export SUBMISSION_DIR={SANDBOX_SUBMISSION_DIR}\n"
            f"export CODE_FOLDER=/code\n"
            f"export PYTHONUNBUFFERED=1\n"
            f"export PYTHONDONTWRITEBYTECODE=1\n"
            f'export PYTHONWARNINGS="ignore::FutureWarning"\n'
            f'exec unshare --mount bash -c "'
            f'mount --bind {shlex.quote(repo_dir)}/data/data {SANDBOX_DATA_DIR}; '
            f'mount --bind {shlex.quote(submission_dir)} {SANDBOX_SUBMISSION_DIR}; '
            f'cd {shlex.quote(executor_dir)}; '
            f'exec {shlex.quote(executor_venv_python)} main.py"\n'
        )
        sandbox.fs.upload_file(wrapper_script.encode("utf-8"), wrapper_path)
        run(sandbox, f"chmod +x {wrapper_path}")
        run(
            sandbox,
            f"DSGYM_EXECUTOR_WORKER={i} nohup bash {wrapper_path} > {log_path} 2>&1 & "
            f"echo $! > {pid_path}",
            check=False,
        )
        # Brief pause for the log file to be created and any immediate startup errors to be written.
        time.sleep(1)
        if sandbox.process.exec(f"test -f {log_path}").exit_code != 0:
            print(f"⚠️  Executor {i} log file {log_path} was not created — wrapper may have failed to start.", file=sys.stderr)
        else:
            log_preview = sandbox.process.exec(f"cat {log_path}")
            if log_preview.result:
                print(f"   Executor {i} log preview:\n{log_preview.result}")

    print("⏳ Waiting for executor workers to become ready...")
    failed = []
    for i in range(n):
        port = EXECUTOR_BASE_PORT + i
        if not _wait_for_executor(sandbox, port, timeout=180):
            failed.append(i)
    if failed:
        print(f"❌ Executor(s) {failed} never became ready. Check /tmp/dsgym_executor_*.log.", file=sys.stderr)
        sys.exit(1)
    print(f"✅ All {n} executor workers ready.")

    state["launch_ok"] = True
    state["num_executors"] = n
    state["container_config_path"] = config_path
    save_state(args.state_file, state)
    return state


# --------------------------------------------------------------------------
# Phase: run  (upload creds, start eval async, stream logs live)
# --------------------------------------------------------------------------

def phase_run(sandbox, state: dict, args) -> dict:
    repo_dir = get_repo_dir(sandbox, state, args)
    max_workers = args.max_workers or args.num_executors or state.get("num_executors", 4)

    if args.backend == "gateway":
        credential_check = sandbox.process.exec(
            'test -n "$AI_CODING_GATEWAY_API_KEY" && '
            'test -n "$AI_CODING_GATEWAY_PROJECT"'
        )
        if credential_check.exit_code != 0:
            print(
                "❌ Gateway credentials are not present in this sandbox. "
                "Create it with `--phase create --backend gateway` after exporting "
                "AI_CODING_GATEWAY_API_KEY and AI_CODING_GATEWAY_PROJECT. "
                "Credentials cannot be added to an already-created sandbox by --phase run.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif (
        args.backend == "litellm"
        and sandbox.process.exec('test -n "$OPENAI_API_KEY"').exit_code != 0
    ):
        print(
            "❌ OPENAI_API_KEY is not present in this sandbox. "
            "Create it with OPENAI_API_KEY exported, or pass --openai-api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Credentials are injected as sandbox-level environment variables at
    # creation time, so they never appear in the command string.
    eval_cmd_parts = [
        f"cd {shlex.quote(repo_dir)} &&",
        "$HOME/.local/bin/uv run dsgym eval",
        f"--dataset {shlex.quote(args.dataset)}",
        f"--model {shlex.quote(args.model)}",
        f"--backend {shlex.quote(args.backend)}",
        f"--limit {args.limit}",
        f"--max-workers {max_workers}",
        f"--dags {'true' if args.dags else 'false'}",
        f"--max-turns {args.max_turns}",
        f"--temperature {args.temperature}",
        f"--seed {args.seed}",
        f"--manager-url http://localhost:{MANAGER_PORT}",
        f"--output-dir {shlex.quote(args.output_dir or f'{repo_dir}/evaluation_results')}",
    ]
    if args.dags:
        if args.prompt is not None:
            eval_cmd_parts.append(f"--prompt {shlex.quote(args.prompt)}")
        if args.dag_model:
            eval_cmd_parts.append(f"--dag-model {shlex.quote(args.dag_model)}")
    if args.sample_n is not None:
        eval_cmd_parts.append(f"--sample-n {args.sample_n}")
        eval_cmd_parts.append(f"--sample-seed {args.sample_seed}")
    if args.run_name:
        eval_cmd_parts.append(f"--run-name {shlex.quote(args.run_name)}")
    if args.tag:
        eval_cmd_parts.append(f"--tag {shlex.quote(args.tag)}")
    eval_cmd = " ".join(eval_cmd_parts)

    eval_session_id = f"dsgym-eval-{int(time.time())}"
    print(f"▶️  Starting eval in background session '{eval_session_id}':\n    {eval_cmd}")
    sandbox.process.create_session(eval_session_id)
    cmd_resp = sandbox.process.execute_session_command(
        eval_session_id,
        SessionExecuteRequest(command=eval_cmd, run_async=True),
    )
    command_id = cmd_resp.cmd_id

    print("📡 Streaming stdout/stderr live (Ctrl-C to detach without stopping the run)...\n")
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_pool_status,
        args=(sandbox, stop_event),
        daemon=True,
    )
    monitor.start()

    # Background auto-fetch thread — periodically saves results so they're
    # not lost if the laptop dies or the sandbox gets auto-deleted.
    auto_fetch_interval = getattr(args, "auto_fetch_interval", 1200)
    af_thread = threading.Thread(
        target=_auto_fetch_loop,
        args=(sandbox, state, args, stop_event, auto_fetch_interval),
        daemon=True,
    )
    af_thread.start()

    try:
        _stream_session_logs(
            sandbox,
            eval_session_id,
            command_id,
            first_upload_timeout=args.first_upload_timeout,
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Detached from stream. The eval is still running in the sandbox.")
        print("   Re-attach with:  --phase attach")
        print("   Auto-fetch continues in background — Ctrl-C again to stop it.")
    finally:
        stop_event.set()
        monitor.join(timeout=15)
        af_thread.join(timeout=15)

    final = sandbox.process.get_session_command(eval_session_id, command_id)
    print(f"\n🏁 Eval finished with exit code {final.exit_code}")
    if final.exit_code != 0:
        print("❌ dsgym eval exited non-zero. Check the streamed output above for details.", file=sys.stderr)
        sys.exit(1)

    state["run_ok"] = True
    save_state(args.state_file, state)
    return state


# --------------------------------------------------------------------------
# Phase: stop-run  (kill a running dsgym eval without stopping the sandbox)
# --------------------------------------------------------------------------

def phase_stop_run(sandbox, state: dict, args) -> dict:
    """Kill the running `dsgym eval` process inside the sandbox.

    This is useful when an eval is burning tokens on a broken setup. It does
    NOT stop the manager/executor services or the sandbox itself.
    """
    print("🛑 Killing any running dsgym eval process in the sandbox...")
    run(sandbox, "pkill -f 'dsgym eval' || true", check=False)
    # Also kill any uv-run wrapper that may be running dsgym eval.
    run(sandbox, "pkill -f 'uv run dsgym eval' || true", check=False)
    print("✅ Sent kill signal. The eval may take a few seconds to stop.")
    return state


# --------------------------------------------------------------------------
# Phase: attach  (re-attach to a running eval session's live log stream)
# --------------------------------------------------------------------------

def phase_attach(sandbox, state: dict, args) -> dict:
    """Find the latest running dsgym-eval-* session and re-stream its logs.

    Useful when your laptop died or you Ctrl-C'd the orchestrator but the eval
    is still running inside the sandbox.
    """
    print("🔎 Searching for running eval sessions...")
    sessions = sandbox.process.list_sessions()
    candidates = []
    for s in sessions:
        sid = s.session_id
        if not sid.startswith("dsgym-eval-"):
            continue
        session_info = sandbox.process.get_session(sid)
        for cmd in session_info.commands:
            if cmd.exit_code is None and "dsgym eval" in cmd.command:
                candidates.append((sid, cmd.id, cmd.command))

    if not candidates:
        print("ℹ️  No running dsgym-eval sessions found.")
        print("   The eval may have finished or been killed. Check results with:")
        print("   --phase exec --command 'ls -lt /root/DSGym/evaluation_results/ | head -10'")
        return state

    # Pick the most recent (highest timestamp in session id)
    candidates.sort(key=lambda c: c[0], reverse=True)
    session_id, command_id, command = candidates[0]
    print(f"📡 Attaching to session '{session_id}':")
    print(f"   {command[:120]}...")
    print()

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_pool_status,
        args=(sandbox, stop_event),
        daemon=True,
    )
    monitor.start()
    try:
        _stream_session_logs(
            sandbox,
            session_id,
            command_id,
            first_upload_timeout=args.first_upload_timeout,
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Detached from stream. The eval is still running in the sandbox.")
        print("   Re-attach with: --phase attach")
    finally:
        stop_event.set()
        monitor.join(timeout=15)

    final = sandbox.process.get_session_command(session_id, command_id)
    print(f"\n🏁 Eval finished with exit code {final.exit_code}")
    return state


# --------------------------------------------------------------------------
# Phase: exec  (run a one-off shell command inside the sandbox)
# --------------------------------------------------------------------------

def phase_exec(sandbox, state: dict, args) -> dict:
    repo_dir = get_repo_dir(sandbox, state, args)
    cwd = args.exec_cwd or repo_dir
    cmd = args.command
    if not cmd:
        print("❌ --command is required for --phase exec.", file=sys.stderr)
        sys.exit(1)
    run(sandbox, cmd, cwd=cwd)
    return state


def _monitor_pool_status(sandbox, stop_event: threading.Event, interval: int = 10) -> None:
    """Background thread: poll manager /status and print a concise pool summary.

    This is purely informational; it does not affect the eval. It prints
    one line per poll showing how many executors are busy/available.
    """
    while not stop_event.is_set():
        try:
            resp = sandbox.process.exec(
                f"curl -sf http://localhost:{MANAGER_PORT}/status",
                timeout=10,
            )
            if resp.exit_code == 0:
                data = json.loads(resp.result)
                total = data.get("total_containers", 0)
                allocated = data.get("allocated_containers", 0)
                available = data.get("available_containers", 0)
                ids = data.get("allocated_container_ids", [])
                msg = (
                    f"🖥️  Pool: {allocated}/{total} busy, {available} available"
                )
                if ids:
                    msg += f"  (busy ids: {ids})"
                print(f"\n{msg}", flush=True)
        except Exception:
            # Fail silently; the live log stream is the primary progress source.
            pass
        # Wait for the interval, but allow stop_event to wake us early.
        stop_event.wait(interval)


def _auto_fetch_loop(sandbox, state, args, stop_event: threading.Event, interval: int) -> None:
    """Background thread: periodically fetch results so they survive laptop disconnects.

    GPU sandboxes are ephemeral — if the sandbox is stopped or auto-deleted,
    all results are lost. This thread fetches every `interval` seconds so
    the latest results are always on the laptop.
    """
    fetch_count = 0
    while not stop_event.is_set():
        try:
            before_sync = state.get("last_volume_sync")
            path = _do_fetch(sandbox, state, args, verbose=False)
            if path:
                fetch_count += 1
                print(f"\n💾 [auto-fetch #{fetch_count}] Results saved to {path}", flush=True)
            elif state.get("last_volume_sync") != before_sync:
                fetch_count += 1
                print(
                    f"\n💾 [auto-fetch #{fetch_count}] Results synced to volume "
                    f"'{KAGGLE_DATA_VOLUME_NAME}' (laptop download skipped)",
                    flush=True,
                )
        except Exception:
            pass  # Fail silently; the eval is the primary concern
        stop_event.wait(interval)


def _stream_session_logs(
    sandbox,
    session_id: str,
    command_id: str,
    *,
    first_upload_timeout: int = 3600,
) -> None:
    """Poll-based live tail of a session command's combined stdout/stderr.

    Uses the simple sync get_session_command_logs (full output each call) and
    prints only the new suffix, so it works without an asyncio event loop.

    Aborts early if the first challenge never reaches Kaggle upload — a sign
    that code execution or executor allocation is broken.
    """
    printed = 0
    eval_started_at: float | None = None
    first_upload_seen = False
    upload_marker = "Upload complete"

    while True:
        logs = sandbox.process.get_session_command_logs(session_id, command_id)
        output = logs.output or ""
        if len(output) > printed:
            new_chunk = output[printed:]
            sys.stdout.write(new_chunk)
            sys.stdout.flush()
            printed = len(output)

            if eval_started_at is None and "Starting evaluation" in output:
                eval_started_at = time.time()
                print(
                    f"\n👀 Health check: waiting for first challenge "
                    f"'✅ {upload_marker}' (timeout {first_upload_timeout}s)...",
                    flush=True,
                )

            if upload_marker in new_chunk:
                first_upload_seen = True
                print("\n✅ First challenge upload confirmed — executor pipeline healthy.", flush=True)

        if (
            eval_started_at is not None
            and not first_upload_seen
            and time.time() - eval_started_at > first_upload_timeout
        ):
            print(
                f"\n❌ No '✅ {upload_marker}' within {first_upload_timeout}s for the first "
                "challenge. Code execution or executor allocation is likely broken.",
                file=sys.stderr,
            )
            print("🛑 Killing the eval run...", file=sys.stderr)
            sandbox.process.exec("pkill -f 'dsgym eval' || true", check=False)
            sandbox.process.exec("pkill -f 'uv run dsgym eval' || true", check=False)
            sys.exit(1)

        cmd = sandbox.process.get_session_command(session_id, command_id)
        if cmd.exit_code is not None:
            # Drain any final output emitted between the last two polls.
            logs = sandbox.process.get_session_command_logs(session_id, command_id)
            output = logs.output or ""
            if len(output) > printed:
                sys.stdout.write(output[printed:])
                sys.stdout.flush()
            break
        time.sleep(3)


# --------------------------------------------------------------------------
# Phase: fetch  (archive + download results)
# --------------------------------------------------------------------------

def phase_fetch(sandbox, state: dict, args) -> dict:
    _do_fetch(sandbox, state, args, verbose=True)
    return state


def _results_persisted(state: dict, sandbox_id: str) -> bool:
    """True if results for this sandbox were synced to the laptop or volume."""
    if state.get("sandbox_id") != sandbox_id:
        return False
    return bool(state.get("last_fetch") or state.get("last_volume_sync"))


def _sync_results_to_volume(
    sandbox,
    repo_dir: str,
    state: dict,
    args,
    verbose: bool = False,
) -> bool:
    """Copy evaluation_results/ to the persistent volume (independent of tar).

    This is the primary safety net for GPU sandboxes: it runs even when
    archiving or laptop download fails.
    """
    results_dir = f"{repo_dir.rstrip('/')}/evaluation_results"
    check = sandbox.process.exec(
        f"test -d {shlex.quote(results_dir)} && "
        f"test -n \"$(ls -A {shlex.quote(results_dir)} 2>/dev/null)\"",
        timeout=30,
    )
    if check.exit_code != 0:
        if verbose:
            print("ℹ️  No evaluation_results to sync to volume yet.")
        return False

    vol_dir = f"{KAGGLE_DATA_VOLUME_MOUNT}/evaluation_results"
    if verbose:
        print(
            f"💾 Syncing evaluation_results/ -> {vol_dir} "
            f"on volume '{KAGGLE_DATA_VOLUME_NAME}'..."
        )
    resp = sandbox.process.exec(
        f"mkdir -p {shlex.quote(vol_dir)} && "
        f"cp -a {shlex.quote(results_dir)}/* {shlex.quote(vol_dir)}/ 2>/dev/null || true",
        timeout=600,
    )
    if resp.exit_code != 0:
        if verbose:
            print(f"⚠️  Volume sync failed (exit {resp.exit_code})")
        return False

    count_resp = sandbox.process.exec(
        f"find {shlex.quote(vol_dir)} -maxdepth 1 -type f 2>/dev/null | wc -l",
        timeout=30,
    )
    count = (count_resp.result or "").strip() if count_resp.exit_code == 0 else "?"
    state["last_volume_sync"] = time.time()
    save_state(args.state_file, state)
    if verbose:
        print(f"✅ Volume sync complete ({count} files on '{KAGGLE_DATA_VOLUME_NAME}').")
    return True


def _do_fetch(sandbox, state: dict, args, verbose: bool = True) -> str:
    """Fetch results from the sandbox. Returns the local archive path, or None.

    Always attempts to sync evaluation_results/ to the persistent volume first.
    Laptop download (tar + extract) is best-effort and may fail independently.
    """
    repo_dir = get_repo_dir(sandbox, state, args)
    volume_ok = _sync_results_to_volume(sandbox, repo_dir, state, args, verbose=verbose)

    remote_archive = f"/tmp/dsgym_results_{int(time.time())}.tar.gz"
    if verbose:
        print(
            f"📦 Archiving evaluation_results/ (+ submissions/) inside the sandbox "
            f"-> {remote_archive} ..."
        )
    resp = sandbox.process.exec(
        f"cd {shlex.quote(repo_dir)} && tar czf {remote_archive} "
        f"evaluation_results $(test -d submissions && echo submissions)",
        timeout=900,
    )
    if resp.exit_code != 0:
        if verbose:
            print(f"⚠️  Archive failed (exit {resp.exit_code})")
            if volume_ok:
                print(
                    f"   Results are on volume '{KAGGLE_DATA_VOLUME_NAME}' — "
                    "recover with a volume-mounted sandbox if needed."
                )
        return None

    local_dir = Path(args.local_results_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_dir / Path(remote_archive).name

    if verbose:
        print(f"⬇️  Downloading to {local_archive} ...")
    try:
        sandbox.fs.download_file(remote_archive, str(local_archive))
    except Exception as e:
        if verbose:
            print(f"⚠️  Download failed: {e}")
            if volume_ok:
                print(
                    f"   Results are on volume '{KAGGLE_DATA_VOLUME_NAME}' — "
                    "laptop download can be retried later."
                )
        return None

    if verbose:
        print(f"📂 Extracting into {local_dir} ...")
    try:
        with tarfile.open(local_archive) as tf:
            tf.extractall(local_dir)
    except Exception as e:
        if verbose:
            print(f"⚠️  Extract failed: {e}")
            if volume_ok:
                print(
                    f"   Results are on volume '{KAGGLE_DATA_VOLUME_NAME}' — "
                    "laptop extract can be retried from the downloaded archive."
                )
        return None

    if verbose:
        print(f"✅ Results available at: {local_dir}")
    state["last_fetch"] = str(local_archive)
    save_state(args.state_file, state)
    return str(local_archive)


# --------------------------------------------------------------------------
# Phase: auto-fetch  (periodically fetch results in a loop)
# --------------------------------------------------------------------------

def phase_auto_fetch(sandbox, state: dict, args) -> dict:
    """Periodically fetch results from the sandbox every --auto-fetch-interval seconds.

    Runs forever until interrupted (Ctrl-C) or the sandbox disappears.
    Use this as a background safety net so results are preserved even if
    your laptop dies or the sandbox gets auto-deleted.
    """
    interval = getattr(args, "auto_fetch_interval", 1200)  # default 20 min
    print(f"🔄 Auto-fetch every {interval}s ({interval // 60} min). Ctrl-C to stop.")
    fetch_count = 0
    while True:
        try:
            before_sync = state.get("last_volume_sync")
            path = _do_fetch(sandbox, state, args, verbose=True)
            if path:
                fetch_count += 1
                print(f"   [auto-fetch #{fetch_count}] Saved to {path}")
            elif state.get("last_volume_sync") != before_sync:
                fetch_count += 1
                print(
                    f"   [auto-fetch #{fetch_count}] Synced to volume "
                    f"'{KAGGLE_DATA_VOLUME_NAME}' (laptop download skipped)"
                )
        except Exception as e:
            print(f"⚠️  Auto-fetch failed: {e}")
        print(f"   Next fetch in {interval // 60} min... (press Ctrl-C to stop)")
        time.sleep(interval)


# --------------------------------------------------------------------------
# Phase: teardown
# --------------------------------------------------------------------------

def phase_teardown(daytona: Daytona, sandbox, state: dict, args) -> dict:
    print("🧹 Stopping manager + executor processes inside the sandbox...")
    _stop_services(sandbox)

    # GPU sandboxes are ephemeral — stopping deletes the filesystem.
    # Sync results to the volume (and laptop if possible) before stopping.
    if not _results_persisted(state, sandbox.id):
        print("⚠️  GPU sandboxes are ephemeral — stopping will DELETE the sandbox.")
        print("📦 Fetching/syncing results before stop...")
        try:
            state = phase_fetch(sandbox, state, args)
        except Exception as e:
            print(f"⚠️  Fetch/sync failed: {e}")
        if not _results_persisted(state, sandbox.id) and not getattr(args, "force", False):
            print("🛑 Refusing to stop without persisting results. Pass --force to skip.", file=sys.stderr)
            return state
    elif not state.get("last_fetch") and state.get("last_volume_sync"):
        print(f"ℹ️  Results synced to volume '{KAGGLE_DATA_VOLUME_NAME}' (laptop fetch not on record).")

    print(f"🛑 Stopping sandbox {sandbox.id} (sandbox will be deleted — GPU sandboxes are ephemeral)...")
    try:
        daytona.stop(sandbox)
    except Exception as e:
        print(f"⚠️  Stop failed: {e}")
    #NOTE: bug below in maybe_delete, commented out for now
    # _maybe_delete(daytona, sandbox, state, args, warn_label="teardown")
    return state


def _maybe_delete(daytona: Daytona, sandbox, state: dict, args, warn_label: str) -> None:
    """Shared delete-guard: stopping is always safe (data survives on disk),
    but deleting destroys the sandbox's filesystem permanently. Only delete
    when explicitly requested, and refuse (unless --force) if this sandbox's
    results were never fetched to the laptop."""
    if not getattr(args, "delete_sandbox", False):
        print(f"ℹ️  Sandbox left STOPPED, not deleted (data preserved). Pass --delete-sandbox "
              f"once you're sure `fetch` succeeded, or re-run `--phase {warn_label}` later.")
        return

    fetched = _results_persisted(state, sandbox.id)
    if not fetched and not getattr(args, "force", False):
        print(
            f"🛑 Refusing to delete sandbox {sandbox.id}: no record of a successful "
            f"`--phase fetch` or volume sync for it in this state file. Results would be LOST.\n"
            f"   Run `--phase fetch` first, or pass --force to delete anyway."
        )
        return

    print(f"🗑️  Deleting sandbox {sandbox.id} ...")
    daytona.delete(sandbox)
    if state.get("sandbox_id") == sandbox.id:
        state.clear()
        save_state(args.state_file, state)


# --------------------------------------------------------------------------
# Phase: kill  (force-STOP by default — this alone stops compute billing and
# preserves the sandbox filesystem/results untouched. Deletion is opt-in via
# --delete-sandbox, and is refused unless a successful `fetch` is on record
# for this sandbox, or --force is passed.)
# --------------------------------------------------------------------------

def _fmt_sandbox_line(sb) -> str:
    err = f"  error={sb.error_reason}" if getattr(sb, "error_reason", None) else ""
    build = ""
    if getattr(sb, "build_info", None):
        build = f"  build={sb.build_info}"
    return (
        f"  - {sb.id}  state={sb.state}  cpu={sb.cpu} mem={sb.memory}GB disk={sb.disk}GB "
        f"gpu={sb.gpu}({sb.gpu_type})  created_at={sb.created_at}  updated_at={sb.updated_at}"
        f"{err}{build}"
    )


# --------------------------------------------------------------------------
# Phase: status  (read-only: view every sandbox in the tenant/org, or just
# ours via --mine, regardless of whether it's still `creating`/
# `pending_build`/`building_snapshot` and blocking a `create` call elsewhere.
# Safe to run at any time, in a second terminal, even while `create` is
# still blocking elsewhere — it only reads, never mutates anything.)
# --------------------------------------------------------------------------

def phase_status(daytona: Daytona, state: dict, args) -> dict:
    from daytona import ListSandboxesQuery

    if args.mine:
        print(f"🔍 Sandboxes labeled {SANDBOX_LABELS} in this org/tenant:")
        sandboxes = list(daytona.list(ListSandboxesQuery(labels=SANDBOX_LABELS)))
    else:
        print("🔍 ALL sandboxes visible in this org/tenant (any project/label):")
        sandboxes = list(daytona.list())

    if not sandboxes:
        print("  (none found)")
        return state

    for sb in sandboxes:
        print(_fmt_sandbox_line(sb))

    tracked_id = state.get("sandbox_id")
    if tracked_id:
        match = next((sb for sb in sandboxes if sb.id == tracked_id), None)
        if match:
            print(f"\nℹ️  Your local state file points at {tracked_id} (state={match.state}).")
        else:
            print(f"\n⚠️  Your local state file points at {tracked_id}, but it wasn't in this "
                  "listing (may have been deleted, or use --mine=false / check target/org).")

    print(
        "\nNote: 'creating' / 'pending_build' / 'building_snapshot' / 'pulling_snapshot' mean "
        "Daytona is still provisioning the sandbox and/or building a snapshot for a non-default "
        "base image (e.g. Image.debian_slim(...)) — this is a one-time cost per unique image and "
        "can take a while, especially combined with GPU node scheduling. It's expected to be much "
        "faster on subsequent `create` calls using the same image once Daytona has cached the "
        "built snapshot."
    )
    return state


def phase_kill(daytona: Daytona, state: dict, args) -> dict:
    sandbox_id = args.sandbox_id or state.get("sandbox_id")
    if not sandbox_id:
        print("❌ No sandbox id available. Pass --sandbox-id <id> or run this against a "
              "state file that has one.", file=sys.stderr)
        sys.exit(1)

    try:
        sandbox = daytona.get(sandbox_id)
    except Exception as e:
        print(f"⚠️  Could not fetch sandbox {sandbox_id} (it may already be gone): {e}")
        if state.get("sandbox_id") == sandbox_id:
            state.clear()
            save_state(args.state_file, state)
        return state

    # GPU sandboxes are ephemeral — stopping deletes the filesystem.
    # Sync results to the volume (and laptop if possible) before stopping.
    if not _results_persisted(state, sandbox_id):
        print("⚠️  GPU sandboxes are ephemeral — stopping will DELETE the sandbox.")
        print("📦 Fetching/syncing results before stop...")
        try:
            state = phase_fetch(sandbox, state, args)
        except Exception as e:
            print(f"⚠️  Fetch/sync failed: {e}")
        if not _results_persisted(state, sandbox_id) and not getattr(args, "force", False):
            print("🛑 Refusing to stop without persisting results. Pass --force to skip.", file=sys.stderr)
            return state
    elif not state.get("last_fetch") and state.get("last_volume_sync"):
        print(f"ℹ️  Results synced to volume '{KAGGLE_DATA_VOLUME_NAME}' (laptop fetch not on record).")

    print(f"🛑 Stopping sandbox {sandbox_id} (state={sandbox.state})... "
          "(sandbox will be deleted — GPU sandboxes are ephemeral)")
    try:
        daytona.stop(sandbox)
    except Exception as e:
        print(f"⚠️  Stop failed: {e}")

    #NOTE: bug below in maybe_delete, commented out for now
    # _maybe_delete(daytona, sandbox, state, args, warn_label="kill")
    print("✅ Done.")
    return state


# --------------------------------------------------------------------------
# Phase: sweep  (safety net independent of any local state file — lists every
# sandbox this script has ever created, by label, in case one was left
# running from a lost/stale state file or a previous crashed run). Stops
# only by default; deletion requires --delete-sandbox AND --force since
# sweep has no way to confirm results were fetched for sandboxes outside
# the local state file.
# --------------------------------------------------------------------------

def phase_sweep(daytona: Daytona, state: dict, args) -> dict:
    from daytona import ListSandboxesQuery

    print(f"🔍 Listing all sandboxes labeled {SANDBOX_LABELS} ...")
    sandboxes = list(daytona.list(ListSandboxesQuery(labels=SANDBOX_LABELS)))

    if not sandboxes:
        print("✅ No matching sandboxes found — nothing to clean up.")
        return state

    print(f"\nFound {len(sandboxes)} sandbox(es):")
    for sb in sandboxes:
        print(f"  - {sb.id}  state={sb.state}  cpu={sb.cpu} mem={sb.memory}GB gpu={sb.gpu} "
              f"created_at={sb.created_at}")

    if not args.yes:
        print("\nRe-run with --yes to STOP all of the sandboxes listed above (this alone "
              "stops billing and preserves their results/filesystems). Add --delete-sandbox "
              "--force on top of --yes only if you're certain results were already fetched "
              "off every one of them, or don't care about their data.")
        return state

    want_delete = getattr(args, "delete_sandbox", False)
    if want_delete and not getattr(args, "force", False):
        print("🛑 Refusing to delete: sweep cannot confirm `fetch` succeeded for these "
              "sandboxes. Add --force to delete anyway, or omit --delete-sandbox to just stop them.")
        want_delete = False

    for sb in sandboxes:
        print(f"\n🛑 Stopping {sb.id} ...")
        try:
            daytona.stop(sb)
        except Exception as e:
            print(f"⚠️  Stop failed for {sb.id}: {e}")
        if want_delete:
            print(f"🗑️  Deleting {sb.id} ...")
            daytona.delete(sb)

    action = "Deleted" if want_delete else "Stopped"
    print(f"\n✅ {action} {len(sandboxes)} sandbox(es).")
    if want_delete and state.get("sandbox_id") in {sb.id for sb in sandboxes}:
        state.clear()
        save_state(args.state_file, state)
    return state


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="all",
                    choices=["create", "preflight", "bootstrap", "launch", "run", "fetch", "auto-fetch", "stop-run", "attach", "exec",
                            "teardown", "kill", "sweep", "status", "all"])
    p.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)

    # Sandbox resources
    p.add_argument("--cpu", type=int, default=16)
    p.add_argument("--memory", type=int, default=64, help="GB RAM")
    p.add_argument("--disk", type=int, default=100, help="GB disk")
    p.add_argument("--gpu-count", type=int, default=2)
    p.add_argument("--gpu-type", default="RTX_4090",
                   help="Comma-separated preference list, e.g. RTX-4090, RTX-5090 (default: RTX-4090)")
    p.add_argument("--adopt-sandbox-id", default=None,
                   help="For --phase create: instead of creating a new sandbox, adopt an "
                        "existing one by id (e.g. one left orphaned in the tenant after a "
                        "daytona.create() call raised locally but the sandbox was actually "
                        "created server-side — check with `--phase status --mine`). Skips "
                        "the create call entirely and just waits for it to reach STARTED.")
    p.add_argument("--create-timeout", type=int, default=1800,
                   help="Seconds to wait for a sandbox to reach STARTED before giving up "
                        "(applies to --phase create's own polling, including after an "
                        "auto-recovered or --adopt-sandbox-id create). Default: 1800 (30 min).")

    # Repo
    p.add_argument("--repo-url", default="https://github.com/Adamwewe/DSGym.git")
    p.add_argument("--branch", default="ai-gateway", #default="feature/scaffolding-with-cuda",
                   help="Git branch to clone into the sandbox (default: feature/scaffolding-with-cuda or feature/daytona-added).")
    p.add_argument("--github-user", default=os.environ.get("GITHUB_USER"))
    p.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"),
                   help="Needed only if the repo is private")

    # Direct-process executors
    p.add_argument("--num-executors", type=int, default=4)
    p.add_argument("--execution-timeout", type=int, default=7200)
    p.add_argument("--mem-limit", default="4G",
                   help="(Deprecated, no-op) Kept for CLI compatibility.")
    p.add_argument("--cpus-per-executor", type=float, default=1.0,
                   help="(Deprecated, no-op) Kept for CLI compatibility.")
    p.add_argument("--mem-reservation", default="1G",
                   help="(Deprecated, no-op) Kept for CLI compatibility.")
    p.add_argument("--allocation-timeout", type=int, default=3600)
    p.add_argument("--first-upload-timeout", type=int, default=3600,
                   help="During --phase run, abort if the first challenge never prints "
                        "'✅ Upload complete' within this many seconds after evaluation "
                        "starts (signals broken code execution / executor allocation). "
                        "Default: 3600 (1 hour).")

    # Eval
    p.add_argument("--dataset", default="dspredict-sampled",
                    choices=["dspredict-easy", "dspredict-hard", "dspredict-sampled"],
                    help="Default is 'dspredict-sampled' (~34 lighter-weight competitions) to keep the "
                         "smoke test's Kaggle data download small. 'dspredict-hard'/'dspredict-easy' pull "
                         "the full ~53-competition set and need a much bigger --disk.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--backend", default="litellm",
                   choices=["gateway", "litellm", "vllm", "sglang"],
                   help="Inference backend passed through to dsgym eval (default: litellm). "
                        "Select the same backend during --phase create so its credentials "
                        "are provisioned in the sandbox.")
    p.add_argument("--limit", type=int, default=1, help="Number of samples to evaluate (adjustable)")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Defaults to --num-executors so workers == executor pool size")
    p.add_argument("--dags", type=lambda x: str(x).lower() not in ("false", "0", "no"),
                    default=True, metavar="BOOL",
                    help="Enable iterative DAG-guided reasoning (DSPredict). Accepts true/false. "
                         "Mirrors dsgym eval's own --dags flag.")
    p.add_argument("--prompt", default="adaptive-revision", metavar="VARIANT",
                   help="Optional DAG system prompt variant (e.g. adaptive-revision, reasoning-only). "
                        "Only passed to dsgym eval when --dags is true. If your dsgym version does not "
                        "support --prompt, omit this flag.")
    p.add_argument("--dag-model", default=None,
                   help="Model used for DAG generation (passed through to dsgym eval --dag-model).")
    p.add_argument("--sample-n", type=int, default=None, metavar="N",
                   help="Randomly sample N tasks from the source split before evaluating "
                        "(passed through to dsgym eval --sample-n).")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="Random seed for --sample-n (default: 42).")
    p.add_argument("--run-name", default=None, help="Custom run name for output files.")
    p.add_argument("--output-dir", default=None,
                   help="Directory for dsgym eval output inside the sandbox (default: <repo>/evaluation_results).")
    p.add_argument("--max-turns", type=int, default=15, help="Maximum agent turns per sample.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for LLM generation reproducibility (default: 42). Passed to dsgym eval --seed.")
    p.add_argument("--tag", default=None,
                   help="Optional tag to label this run (e.g. 'no-cuda', 'baseline'), included in "
                        "dsgym eval's JSON/CSV output for cross-run filtering.")
    p.add_argument("--openai-api-key", default=None, help="Defaults to $OPENAI_API_KEY")

    # Results
    p.add_argument("--local-results-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--auto-fetch-interval", type=int, default=1200,
                   help="Auto-fetch interval in seconds (default: 1200 = 20 min). "
                        "Used by the background auto-fetch thread in --phase run, and by --phase auto-fetch.")

    # Teardown / kill / sweep
    p.add_argument("--delete-sandbox", action="store_true",
                   help="For --phase teardown/kill/sweep: also DELETE the sandbox (destroys its "
                        "filesystem permanently) in addition to stopping it. Refused unless a "
                        "successful `fetch` is on record (or --force is passed) — stopping alone "
                        "already halts billing and preserves results.")
    p.add_argument("--force", action="store_true",
                   help="Allow --delete-sandbox to proceed even without a recorded successful "
                        "`fetch` for the sandbox(es) being deleted. Use with care — this can "
                        "permanently destroy unretrieved results.")
    p.add_argument("--sandbox-id", default=None,
                   help="For --phase kill: target a specific sandbox id directly, bypassing the "
                        "state file (useful if it was lost or you're killing a different run's sandbox).")
    p.add_argument("--yes", action="store_true",
                   help="For --phase sweep: actually STOP every matching sandbox found (billing "
                        "stops, results preserved). Without this flag, sweep only lists them "
                        "(dry-run). Combine with --delete-sandbox --force to also delete them.")
    p.add_argument("--mine", action="store_true",
                   help="For --phase status: only show sandboxes labeled for this project "
                        "(same filter as --phase sweep). Without this flag, status shows "
                        "EVERY sandbox visible in your org/tenant, which is useful for "
                        "spotting unrelated or forgotten sandboxes too.")
    p.add_argument("--command", default=None,
                   help="For --phase exec: the shell command to run inside the sandbox.")
    p.add_argument("--exec-cwd", default=None,
                   help="For --phase exec: working directory inside the sandbox "
                        "(default: repo dir from state).")

    return p


PHASE_ORDER = ["create", "preflight", "bootstrap", "launch", "run", "fetch"]


def main():
    args = build_arg_parser().parse_args()
    daytona = Daytona()
    state = load_state(args.state_file)

    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    for phase in phases:
        print(f"\n{'=' * 70}\nPHASE: {phase}\n{'=' * 70}")
        if phase == "create":
            state = phase_create(daytona, state, args)
        elif phase == "preflight":
            state = phase_preflight(get_sandbox(daytona, state), state, args)
        elif phase == "bootstrap":
            state = phase_bootstrap(get_sandbox(daytona, state), state, args)
        elif phase == "launch":
            state = phase_launch(get_sandbox(daytona, state), state, args)
        elif phase == "run":
            state = phase_run(get_sandbox(daytona, state), state, args)
        elif phase == "fetch":
            state = phase_fetch(get_sandbox(daytona, state), state, args)
        elif phase == "auto-fetch":
            state = phase_auto_fetch(get_sandbox(daytona, state), state, args)
        elif phase == "teardown":
            state = phase_teardown(daytona, get_sandbox(daytona, state), state, args)
        elif phase == "kill":
            state = phase_kill(daytona, state, args)
        elif phase == "sweep":
            state = phase_sweep(daytona, state, args)
        elif phase == "status":
            state = phase_status(daytona, state, args)
        elif phase == "exec":
            state = phase_exec(get_sandbox(daytona, state), state, args)
        elif phase == "stop-run":
            state = phase_stop_run(get_sandbox(daytona, state), state, args)
        elif phase == "attach":
            state = phase_attach(get_sandbox(daytona, state), state, args)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
