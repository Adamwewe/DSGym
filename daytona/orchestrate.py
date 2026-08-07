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
  run        Write KAGGLE_TOKEN to ~/.kaggle/access_token + push OPENAI_API_KEY,
             start `dsgym eval` in a background Daytona session, and stream its
             stdout/stderr live to this terminal until it finishes.
  fetch      Tar evaluation_results/ (+ submissions/) inside the sandbox and
             download the archive to --local-results-dir.
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

State (sandbox id, repo dir, etc.) is persisted to --state-file (default:
daytona/.daytona_state.json) so phases can be re-run independently without
re-creating the sandbox each time.

Credentials (never committed, never baked into images):
  DAYTONA_API_KEY   Required. Auth for the Daytona SDK itself.
  OPENAI_API_KEY    Required. Passed into the sandbox as a secret env var
                    for the gpt-4o agent under evaluation.
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
# Applied to every sandbox this script creates, so `sweep` can find (and
# kill) them later even without a local state file.
SANDBOX_LABELS = {"project": "dsgym", "purpose": "dspredict-gpu-smoketest"}
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
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not set locally and --openai-api-key not passed. "
              "The sandbox needs it at creation time so `dsgym eval` can read it later.", file=sys.stderr)
        sys.exit(1)

    print(f"🚀 Creating Daytona GPU sandbox (cpu={args.cpu}, memory={args.memory}GB, "
          f"disk={args.disk}GB, gpu={args.gpu_count}, gpu_type={gpu_types or 'any available'})...")

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
        env_vars={"OPENAI_API_KEY": openai_key},
        auto_delete_interval=0,
        auto_stop_interval=0,
        labels=SANDBOX_LABELS,
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
        "build-essential libgomp1 curl ca-certificates gnupg git unzip "
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
        f"$HOME/.local/bin/uv venv --python 3.12 {shlex.quote(venv_dir)}"
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
    """Install the executor-kaggle Python dependencies into the sandbox's uv venv.

    The executor image is based on pytorch/pytorch which ships CUDA-enabled torch.
    We install all of requirements.kaggle.txt first, then force-reinstall torch from
    the CUDA 12.6 wheels so the GPU build wins (requirements.kaggle.txt itself lists
    'torch', which would otherwise pull the CPU wheel from PyPI).
    """
    print("🔧 Installing executor-kaggle Python dependencies (force-reinstall to match requirements.kaggle.txt)...")
    venv_python = get_venv_python(repo_dir)
    run(sandbox, (
        f"$HOME/.local/bin/uv pip install --python {shlex.quote(venv_python)} --force-reinstall -r "
        f"{shlex.quote(repo_dir)}/executors/container_images/kaggle_image/requirements.kaggle.txt"
    ), cwd=repo_dir, timeout=600)

    print("🔧 Upgrading torch to CUDA 12.6 wheels (force-reinstall over the CPU wheel)...")
    run(sandbox, (
        f"$HOME/.local/bin/uv pip install --python {shlex.quote(venv_python)} --force-reinstall --index-url "
        "https://download.pytorch.org/whl/cu126 torch"
    ), cwd=repo_dir, timeout=600)

    print("🔎 Verifying installed PyTorch can see the GPU...")
    venv_python = get_venv_python(repo_dir)
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


def _stop_services(sandbox) -> None:
    """Kill any manager/executor processes started by this orchestrator."""
    print("🧹 Stopping manager + executor processes...")
    sandbox.process.exec(
        "pkill -f 'DSGYM_EXECUTOR_WORKER' || true; "
        "pkill -f 'executors/manager/main.py' || true; "
        "pkill -f 'executors/container_images/kaggle_image/main.py' || true; "
        "rm -f /tmp/dsgym_executor_*.pid /tmp/dsgym_manager.pid"
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

    download_script = {
        "dspredict-sampled": "download_dspredict_data.sh",
        "dspredict-hard": "download_dspredict_data_full.sh",
        "dspredict-easy": "download_dspredict_data_full.sh",
    }[args.dataset]
    print(f"📊 Downloading Kaggle competition data via {download_script} "
          f"(skips files already present, safe to re-run)...")
    run(sandbox, f"PATH=$HOME/.local/bin:$PATH $HOME/.local/bin/uv run kaggle --version", cwd=repo_dir)
    run(sandbox, (
        f"PATH=$HOME/.local/bin:$PATH "
        f"bash {download_script}"
    ), cwd=repo_dir, timeout=3600)

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
        f"nohup {shlex.quote(manager_venv_python)} main.py > /tmp/dsgym_manager.log 2>&1 &"
    )
    run(sandbox, manager_cmd, check=False)
    run(sandbox, "sleep 2 && echo $! > /tmp/dsgym_manager.pid")

    _wait_for_manager(sandbox)

    print(f"🚀 Starting {n} executor-kaggle worker(s) with mount-namespace isolation...")
    venv_python = get_venv_python(repo_dir)
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
            f'exec {shlex.quote(venv_python)} main.py"\n'
        )
        sandbox.fs.upload_file(wrapper_script.encode("utf-8"), wrapper_path)
        run(sandbox, f"chmod +x {wrapper_path}")
        run(sandbox, f"DSGYM_EXECUTOR_WORKER={i} nohup bash {wrapper_path} > {log_path} 2>&1 &", check=False)
        run(sandbox, f"echo $! > {pid_path}")
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

    # OPENAI_API_KEY was injected as a sandbox-level env var at `create` time
    # (see phase_create), so it's already in this shell's environment here —
    # no need to embed the raw key in this command string.
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
    try:
        _stream_session_logs(sandbox, eval_session_id, command_id)
    finally:
        stop_event.set()
        monitor.join(timeout=15)

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


def _stream_session_logs(sandbox, session_id: str, command_id: str, stop_event: threading.Event = None) -> None:
    """Poll-based live tail of a session command's combined stdout/stderr.

    Uses the simple sync get_session_command_logs (full output each call) and
    prints only the new suffix, so it works without an asyncio event loop.
    """
    printed = 0
    while True:
        logs = sandbox.process.get_session_command_logs(session_id, command_id)
        output = logs.output or ""
        if len(output) > printed:
            sys.stdout.write(output[printed:])
            sys.stdout.flush()
            printed = len(output)

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
    repo_dir = get_repo_dir(sandbox, state, args)
    remote_archive = f"/tmp/dsgym_results_{int(time.time())}.tar.gz"
    print(f"📦 Archiving evaluation_results/ (+ submissions/) inside the sandbox -> {remote_archive} ...")
    run(sandbox, (
        f"cd {repo_dir} && tar czf {remote_archive} "
        f"evaluation_results $(test -d submissions && echo submissions)"
    ))

    local_dir = Path(args.local_results_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_dir / Path(remote_archive).name

    print(f"⬇️  Downloading to {local_archive} ...")
    sandbox.fs.download_file(remote_archive, str(local_archive))

    print(f"📂 Extracting into {local_dir} ...")
    with tarfile.open(local_archive) as tf:
        tf.extractall(local_dir)

    print(f"✅ Results available at: {local_dir}")
    state["last_fetch"] = str(local_archive)
    save_state(args.state_file, state)

    print(
        "\n💰 Reminder: the sandbox is still RUNNING and billing until you stop it.\n"
        f"   Run:  daytona/.venv/bin/python daytona/orchestrate.py --phase kill --state-file {args.state_file}\n"
        "   This only STOPS it (billing stops, results/filesystem preserved) — nothing is\n"
        "   deleted unless you separately pass --delete-sandbox (and it will refuse to delete\n"
        "   unless this fetch is on record, since you just ran it, or --force)."
    )
    return state


# --------------------------------------------------------------------------
# Phase: teardown
# --------------------------------------------------------------------------

def phase_teardown(daytona: Daytona, sandbox, state: dict, args) -> dict:
    print("🧹 Stopping manager + executor processes inside the sandbox...")
    _stop_services(sandbox)
    print(f"🛑 Stopping sandbox {sandbox.id} (billing stops; results/filesystem preserved)...")
    try:
        daytona.stop(sandbox)
    except Exception as e:
        print(f"⚠️  Stop failed: {e}")
    _maybe_delete(daytona, sandbox, state, args, warn_label="teardown")
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

    fetched = state.get("last_fetch") and state.get("sandbox_id") == sandbox.id
    if not fetched and not getattr(args, "force", False):
        print(
            f"🛑 Refusing to delete sandbox {sandbox.id}: no record of a successful "
            f"`--phase fetch` for it in this state file. Results would be LOST.\n"
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

    print(f"🛑 Stopping sandbox {sandbox_id} (state={sandbox.state})... "
          "(compute billing stops here; the filesystem/results are preserved)")
    try:
        daytona.stop(sandbox)
    except Exception as e:
        print(f"⚠️  Stop failed: {e}")

    _maybe_delete(daytona, sandbox, state, args, warn_label="kill")
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
                    choices=["create", "preflight", "bootstrap", "launch", "run", "fetch", "stop-run", "exec",
                            "teardown", "kill", "sweep", "status", "all"])
    p.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)

    # Sandbox resources
    p.add_argument("--cpu", type=int, default=4)
    p.add_argument("--memory", type=int, default=32, help="GB RAM")
    p.add_argument("--disk", type=int, default=100, help="GB disk")
    p.add_argument("--gpu-count", type=int, default=1)
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
    p.add_argument("--branch", default="feature/scaffolding-with-cuda",
                   help="Git branch to clone into the sandbox (default: feature/scaffolding-with-cuda).")
    p.add_argument("--github-user", default=os.environ.get("GITHUB_USER"))
    p.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"),
                   help="Needed only if the repo is private")

    # Direct-process executors
    p.add_argument("--num-executors", type=int, default=4)
    p.add_argument("--execution-timeout", type=int, default=1200)
    p.add_argument("--mem-limit", default="4G",
                   help="(Deprecated, no-op) Kept for CLI compatibility.")
    p.add_argument("--cpus-per-executor", type=float, default=1.0,
                   help="(Deprecated, no-op) Kept for CLI compatibility.")
    p.add_argument("--mem-reservation", default="1G",
                   help="(Deprecated, no-op) Kept for CLI compatibility.")
    p.add_argument("--allocation-timeout", type=int, default=3600)

    # Eval
    p.add_argument("--dataset", default="dspredict-sampled",
                    choices=["dspredict-easy", "dspredict-hard", "dspredict-sampled"],
                    help="Default is 'dspredict-sampled' (~34 lighter-weight competitions) to keep the "
                         "smoke test's Kaggle data download small. 'dspredict-hard'/'dspredict-easy' pull "
                         "the full ~53-competition set and need a much bigger --disk.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--backend", default="litellm",
                   choices=["litellm", "vllm", "sglang"],
                   help="Inference backend passed through to dsgym eval (default: litellm).")
    p.add_argument("--limit", type=int, default=5, help="Number of samples to evaluate (adjustable)")
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
    p.add_argument("--tag", default=None,
                   help="Optional tag to label this run (e.g. 'no-cuda', 'baseline'), included in "
                        "dsgym eval's JSON/CSV output for cross-run filtering.")
    p.add_argument("--openai-api-key", default=None, help="Defaults to $OPENAI_API_KEY")

    # Results
    p.add_argument("--local-results-dir", default=str(SCRIPT_DIR / "results"))

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

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
