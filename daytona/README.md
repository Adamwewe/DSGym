# Running DSGym on Daytona (GPU sandbox smoke test)

Orchestrates a DSPredict evaluation against N `executor-kaggle` workers + 1
manager, all running as direct processes inside a single Daytona GPU sandbox
(no Docker-in-Docker). The script lives in this folder and drives the
sandbox from your laptop with the Daytona Python SDK.

## Why a separate venv

`daytona/requirements.txt` installs the `daytona` SDK into its own venv, kept
completely separate from DSGym's own pinned dependencies (`pyproject.toml` /
`uv.lock`). Do not `uv pip install daytona` into the repo's main `.venv` —
it drags in newer/conflicting versions of `pydantic` and other shared deps.

```bash
cd daytona
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## Credentials

Export these locally before running (never commit them, never bake them
into an image):

| Variable | Purpose |
|---|---|
| `DAYTONA_API_KEY` | Auth for the Daytona SDK (sandbox create/exec/fs). |
| `OPENAI_API_KEY` | Passed into the sandbox as an env var for the gpt-4o agent under evaluation. |
| `KAGGLE_TOKEN` | Modern Kaggle access token (starts with `KGAT_...`), written to `~/.kaggle/access_token` inside the sandbox — required for DSPredict data download and submission scoring. |
| `GITHUB_TOKEN` / `GITHUB_USER` | Only needed if `--repo-url` points at a private repo. |

## Quick start

Once the environment is setup, run the bash script to spin up everything for the first run

```
./run_experiments.sh
```

## Monitoring runs

While an eval streams to your laptop, watch for healthy per-challenge progress:

- **Container allocation** — `🔧 Allocated container: N` and pool status (`🖥️  Pool: X/Y busy`)
- **Code execution** — agent `<python>` blocks producing output in the stream
- **Result summaries** — `💾 Saved prediction to: ...` followed by medal/score summaries
- **Kaggle upload** — `✅ Upload complete: ...` after each submission is scored

**Early abort rule:** if the **first** challenge never prints `✅ Upload complete`
within ~1 hour of `🏃 Starting evaluation`, code execution or executor allocation
is likely broken — kill the run (`--phase stop-run`) and relaunch the executor pool
(`--phase launch`) before retrying.

The orchestrator enforces this automatically during `--phase run` via
`--first-upload-timeout` (default 3600s).

## Usage

Run everything end-to-end (create → preflight → bootstrap → launch → run → fetch):

```bash
export DAYTONA_API_KEY=...
export OPENAI_API_KEY=...
export KAGGLE_TOKEN=KGAT_...
.venv/bin/python orchestrate.py --phase all \
  --num-executors 4 --limit 5 --model gpt-4o
```

Or drive it phase-by-phase (useful when iterating, since sandbox creation
and dependency installs are slow and phases are re-runnable independently via
the persisted state file `.daytona_state.json`):

```bash
.venv/bin/python orchestrate.py --phase create
.venv/bin/python orchestrate.py --phase preflight   # verify GPU + mount namespaces
.venv/bin/python orchestrate.py --phase bootstrap
.venv/bin/python orchestrate.py --phase launch
.venv/bin/python orchestrate.py --phase run --limit 5     # --limit is adjustable on every run
.venv/bin/python orchestrate.py --phase fetch
.venv/bin/python orchestrate.py --phase kill              # stop billing immediately once results are downloaded
```

Debug inside the running sandbox without SSH:

```bash
# Check manager health
.venv/bin/python orchestrate.py --phase exec --command "curl -s http://localhost:5000/health"

# Read a manager/executor log
.venv/bin/python orchestrate.py --phase exec --command "cat /tmp/dsgym_manager.log"
.venv/bin/python orchestrate.py --phase exec --command "cat /tmp/dsgym_executor_000.log"

# List running processes
.venv/bin/python orchestrate.py --phase exec --command "ps aux | grep -E 'main|dsgym'"

# Stop a run that is wasting tokens on a broken setup
.venv/bin/python orchestrate.py --phase stop-run

# Kill everything (dsgym eval + manager + executors) and relaunch fresh
.venv/bin/python orchestrate.py --phase stop-run
.venv/bin/python orchestrate.py --phase launch --num-executors 4 --execution-timeout 7200

# Re-attach to a running eval's live log stream (e.g. after laptop restart)
.venv/bin/python orchestrate.py --phase attach
```

## Workflow

Typical lifecycle (each phase uses the state file so you can re-run or resume):

```bash
# 1. One-time provisioning + checks
.venv/bin/python orchestrate.py --phase create
.venv/bin/python orchestrate.py --phase preflight

# 2. One-time setup (clone, install deps, download Kaggle data)
.venv/bin/python orchestrate.py --phase bootstrap

# 3. Start manager + executor workers
.venv/bin/python orchestrate.py --phase launch

# 4. Run the evaluation (can be repeated with different --limit/--model/etc)
.venv/bin/python orchestrate.py --phase run --limit 5
.venv/bin/python orchestrate.py --phase run --limit 50

# 5. Pull results back
.venv/bin/python orchestrate.py --phase fetch

# 6. Stop billing (keeps the filesystem until you explicitly delete it)
.venv/bin/python orchestrate.py --phase kill
```

Important rules:

- **Do not keep `launch` open in another terminal.** It starts background
  processes inside the sandbox and exits. The manager/executors keep running
  until you run `--phase teardown` or `--phase kill`.
- **You can re-run `run` many times** after a single `launch` — each `run`
  creates a fresh Daytona session so it will never conflict with a previous
  attempt.
- **`launch` only works while the sandbox exists.** If you run `--phase kill`
  (or delete the sandbox), you must go back to `--phase create` before you can
  `launch` again.

## Configuration

All flags are optional unless marked **Required**.

### Sandbox resources

| Flag | Default | Description |
|---|---|---|
| `--cpu` | 16 | Sandbox vCPUs |
| `--memory` | 64 | Sandbox RAM (GiB) |
| `--disk` | 100 | Sandbox disk (GiB) |
| `--gpu-count` | 2 | Number of GPUs |
| `--gpu-type` | `RTX_4090` | Comma-separated GPU preference list, e.g. `RTX_4090,RTX_5090` |
| `--create-timeout` | 1800 | Seconds to wait for the sandbox to reach `STARTED` |

### Direct-process executor pool

| Flag | Default | Description |
|---|---|---|
| `--num-executors` | 4 | Number of executor-kaggle workers to start |
| `--execution-timeout` | 7200 | Seconds the executor will wait for a single code execution (2 hours) |
| `--allocation-timeout` | 3600 | Seconds the manager will queue a task waiting for a free worker |
| `--mem-limit` | 4G | Deprecated, kept for CLI compatibility only |
| `--cpus-per-executor` | 1.0 | Deprecated, kept for CLI compatibility only |
| `--mem-reservation` | 1G | Deprecated, kept for CLI compatibility only |

### DSGym eval flags (passed through 1:1)

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `dspredict-sampled` | Dataset to evaluate: `dspredict-sampled`, `dspredict-hard`, `dspredict-easy` |
| `--model` | `gpt-4o` | Model name passed to `dsgym eval --model` |
| `--backend` | `litellm` | Inference backend: `litellm`, `vllm`, `sglang` |
| `--limit` | 5 | Number of samples to evaluate |
| `--max-workers` | `--num-executors` | Number of parallel `dsgym eval` workers |
| `--dags` | `true` | Enable DAG-guided reasoning (`true` or `false`) |
| `--prompt` | — | Optional DAG prompt variant when `--dags true` (e.g. `reasoning-only`). Only passed if explicitly set; omit it if your `dsgym` version does not support `--prompt`. |
| `--dag-model` | — | Model for DAG generation (passed to `dsgym eval --dag-model`) |
| `--sample-n` | — | Randomly sample N tasks before evaluating |
| `--sample-seed` | 42 | Seed for `--sample-n` |
| `--max-turns` | 15 | Maximum agent turns per sample |
| `--temperature` | 0.0 | Sampling temperature |
| `--seed` | 42 | Random seed for LLM generation reproducibility |
| `--run-name` | — | Custom prefix for output files |
| `--tag` | — | Label included in output JSON/CSV for cross-run filtering |
| `--output-dir` | `<repo>/evaluation_results` | Directory inside the sandbox for evaluation outputs |

### Other

| Flag | Default | Description |
|---|---|---|
| `--openai-api-key` | `$OPENAI_API_KEY` | **Required** unless `OPENAI_API_KEY` is exported |
| `KAGGLE_TOKEN` | `$KAGGLE_TOKEN` | **Required** for DSPredict; written to `~/.kaggle/access_token` in the sandbox |
| `--command` | — | Shell command to run inside the sandbox (required for `--phase exec`) |
| `--exec-cwd` | repo root | Working directory for `--phase exec` |
| `--local-results-dir` | `daytona/results` | Local directory where the fetched archive is extracted |
| `--auto-fetch-interval` | 1200 | Auto-fetch interval in seconds (20 min). Used by `--phase run` background thread and `--phase auto-fetch` |
| `--repo-url` | `https://github.com/Adamwewe/DSGym.git` | Git repo cloned into the sandbox |
| `--branch` | `feature/scaffolding-with-cuda` | Git branch to clone |
| `--github-user` / `--github-token` | — | Only needed if `--repo-url` is private |

## Example runs

Small smoke test (default):

```bash
.venv/bin/python orchestrate.py --phase all \
  --num-executors 4 --limit 5
```

Larger DSPredict run with DAGs:

```bash
.venv/bin/python orchestrate.py --phase all \
  --dataset dspredict-hard \
  --num-executors 8 --limit 2 \
  --dags true --prompt adaptive-revision \
  --max-turns 3 --execution-timeout 3600 --allocation-timeout 7200
```

Sample a subset once, then evaluate with different settings on the same sample:

```bash
# First run: create the sample and evaluate with DAGs
.venv/bin/python orchestrate.py --phase all \
  --dataset dspredict-hard --sample-n 10 --sample-seed 42 \
  --num-executors 4 --dags true

# Second run (no re-sampling): use the saved dspredict-sampled split
.venv/bin/python orchestrate.py --phase run --fetch --kill \
  --dataset dspredict-sampled --dags false --num-executors 4
```

Use a different model or backend:

```bash
.venv/bin/python orchestrate.py --phase run \
  --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
  --backend litellm \
  --limit 10
```

## What each phase does

- **create** — creates a GPU sandbox (default: 4 vCPU / 32GB RAM / 100GB disk
  / 1 GPU) using `Image.base("ubuntu:22.04")` with Python and build tools
  pre-installed, and injects `OPENAI_API_KEY` as a sandbox-level env var so
  no later command has to embed the raw key in a shell string.

  **Why `create` can take 10+ minutes the first time:** any non-default
  base image (like `ubuntu:22.04`) isn't one of Daytona's pre-built
  snapshots, so the very first `create` with that image has to go through
  `pending_build` → `building_snapshot` before it can even start
  provisioning the GPU node. This is a one-time cost per unique image —
  once Daytona has cached the built snapshot, subsequent `create` calls
  with the same image are much faster. GPU node scheduling on top of that
  can add more latency than a CPU-only sandbox. Use `--phase status` (see
  below) in a second terminal to see exactly which state it's stuck in
  instead of guessing.

  **If `create` crashes with a 502/gateway error:** this means the SDK's
  own internal wait-for-start hit a transient gateway hiccup — the sandbox
  itself is very likely still alive server-side (check with
  `--phase status --mine`; it usually still shows up mid-build). The
  script auto-recovers from this in-process (finds and adopts the most
  recently created matching sandbox, then polls it itself until it
  reaches `STARTED`, using `--create-timeout`, default 30 min). If it was
  a past run before this recovery existed and you're left with an
  orphaned sandbox not in your state file, adopt it manually instead of
  paying the snapshot-build cost twice:

  ```bash
  .venv/bin/python orchestrate.py --phase status --mine   # find its id
  .venv/bin/python orchestrate.py --phase create --adopt-sandbox-id <id>
  ```

- **status** — read-only, safe to run anytime (including from a second
  terminal while `create` is still blocking elsewhere in the first one).
  Lists every sandbox visible in your org/tenant with its live `state`
  (`creating`, `pending_build`, `building_snapshot`, `pulling_snapshot`,
  `started`, `error`, ...), resources, and `error_reason`/`build_info` if
  present.

- **preflight** — verifies `nvidia-smi` sees the GPU and that the sandbox
  supports Linux mount namespaces (used to give each executor process its
  own isolated `/data` and `/submission` bind mounts). This is the riskiest,
  least-documented step, so it's split out on its own and fails fast with
  diagnostics if it doesn't work.

- **bootstrap** — clones the repo, installs `uv`, runs `uv sync`, then
  installs the Python dependencies for the manager and the executor-kaggle
  worker (including CUDA 12.6 PyTorch). Upgrades `kaggle` to a version that
  supports the modern `~/.kaggle/access_token` format, writes `$KAGGLE_TOKEN`
  to that file, and downloads the Kaggle competition data via the appropriate
  `download_*.sh` script.

  **Kaggle data is stored on a persistent Daytona volume**
  (`dsgym-kaggle-data`) mounted at `/mnt/kaggle-data`. The first `bootstrap`
  downloads all competitions (10–30 min) and copies them to the volume.
  Subsequent sandboxes find the data on the volume and skip the download
  entirely — just symlinks into the repo's `data/data/` directory.

  **The volume also stores `evaluation_results/`** — every `fetch` and
  `auto-fetch` copies results to the volume, so they persist across sandbox
  deletions. When a new sandbox bootstraps, it restores any previous results
  from the volume.

  If the volume is missing or empty, `bootstrap` downloads from scratch.

  **Kaggle download can take 10–30 minutes** (30+ competitions). The
  orchestrator waits for the script to finish before printing any output, so
  the terminal will appear frozen at `bash download_dspredict_data.sh`. To
  check progress from a second terminal:

  ```bash
  .venv/bin/python orchestrate.py --phase exec --command "find /root/DSGym/data/data/dspredict-sampled -maxdepth 2 -type f | wc -l"
  .venv/bin/python orchestrate.py --phase exec --command "ls /root/DSGym/data/data/dspredict-sampled/"
  ```

  The download timeout is 1 hour. Re-running `bootstrap` is safe — the script
  skips competitions that are already downloaded.

- **launch** — generates a `daytona_container_config.json` pointing each
  worker at a distinct `localhost` port, starts the manager process on port
  5000, then starts N executor workers in their own mount namespaces. Each
  worker sees `/data` mapped to `data/data` and `/submission` mapped to a
  per-worker directory under `submissions/`. Waits for the manager's
  `/health` endpoint and each worker's `/ready` endpoint.

- **run** — starts `uv run dsgym eval ...` in a background Daytona session
  and streams combined stdout/stderr live to your terminal until it finishes.
  A background thread also polls the manager's `/status` endpoint every 10
  seconds and prints a concise pool summary (e.g. `2/4 busy, 2 available`) so
  you can see executor utilization without scrolling through the eval log.

- **exec** — runs a one-off shell command inside the sandbox. Useful for
  quick debugging, e.g. `--command "curl -s http://localhost:5000/health"` or
  `--command "cat /tmp/dsgym_manager.log"`. Use `--exec-cwd` to override the
  working directory (default is the repo root).

- **stop-run** — kills a running `dsgym eval` process inside the sandbox
  without stopping the manager/executors or the sandbox. Use this when an eval
  is burning tokens on a broken setup.

- **attach** — re-attaches to a running eval session's live log stream. Finds
  the latest `dsgym-eval-*` session that's still running and re-streams its
  stdout/stderr. Use this when your laptop died or you Ctrl-C'd the orchestrator
  but the eval is still running inside the sandbox.

```bash
.venv/bin/python orchestrate.py --phase attach
```

- **fetch** — `tar czf`s `evaluation_results/` (+ `submissions/`) inside the
  sandbox and downloads the archive to `daytona/results/` on your laptop.

- **auto-fetch** — runs `fetch` in a loop every `--auto-fetch-interval` seconds
  (default 1200 = 20 min). Use as a standalone background safety net:

  ```bash
  # In a separate terminal, runs forever until Ctrl-C:
  .venv/bin/python orchestrate.py --phase auto-fetch
  ```

  This is critical because **GPU sandboxes are ephemeral** — if the sandbox
  is stopped or auto-deleted, all results are lost. Auto-fetch ensures the
  latest results are always on your laptop.

- **teardown** — kills all manager/executor background processes inside the
  sandbox, then STOPS the sandbox (billing stops; filesystem/results
  preserved). Pass `--delete-sandbox` to also delete it — refused unless a
  successful `fetch` is on record for this sandbox, or `--force`.

## Architecture: why no Docker?

Daytona GPU sandboxes are containers themselves. Running Docker inside a
container (Docker-in-Docker) requires nested container privileges that are
blocked in this environment — even a simple `docker run alpine` fails with
`/proc` cross-device link errors. Rather than fight that, this orchestrator
runs the manager and executor workers as direct processes inside the sandbox:

- The manager is just a FastAPI HTTP proxy; it needs no Docker.
- Each executor is just a FastAPI server wrapping a Jupyter kernel; it needs no Docker.
- All communication is HTTP over `localhost`, exactly like the Docker setup.
- The only thing Docker provided was **per-container filesystem isolation**
  for `/data` and `/submission`. That is replaced by **Linux mount namespaces**:
  each worker process starts in its own namespace with `/data` and
  `/submission` bind-mounted to per-worker directories, so the agent still
  sees the same hardcoded paths (`/submission/submission.csv`) as in the
  original Docker configuration.

## Stopping billing (important)

Sandboxes bill while running, regardless of whether `dsgym eval` is still
active. **Stopping** a sandbox halts billing and leaves its filesystem (and
therefore any results you haven't fetched yet) completely intact — it's
always safe. **Deleting** permanently destroys that filesystem, so this
script defaults to stop-only everywhere and requires you to opt in to
deletion explicitly:

- **`--phase kill`** — the fast path. STOPS the sandbox from the current
  state file (or `--sandbox-id <id>` if you don't have/trust the state
  file). This alone is enough to stop billing.

  **GPU sandboxes are ephemeral** — stopping them permanently deletes the
  filesystem. `kill` and `teardown` will **auto-fetch results before
  stopping** if no `fetch` is on record. Pass `--force` to skip the
  auto-fetch (results will be lost).

  ```bash
  .venv/bin/python orchestrate.py --phase kill
  # or, if the state file is stale/missing:
  .venv/bin/python orchestrate.py --phase kill --sandbox-id sandbox_abc123
  ```

  Add `--delete-sandbox` to also delete it once you're done with it for
  good — this is refused unless a successful `--phase fetch` is on record
  for that sandbox in the state file (add `--force` to override, e.g. if
  you deliberately don't care about its results):

  ```bash
  .venv/bin/python orchestrate.py --phase kill --delete-sandbox
  ```

- **`--phase sweep`** — the safety net. Lists **every** sandbox this script
  has ever created (matched by label, independent of any state file), so you
  can catch one left running from a crashed run or a lost state file. Dry
  run by default; add `--yes` to STOP all of them (results preserved):

  ```bash
  .venv/bin/python orchestrate.py --phase sweep          # dry run — just lists
  .venv/bin/python orchestrate.py --phase sweep --yes     # stops all of them
  ```

  Because `sweep` operates on sandboxes outside the local state file, it
  has no way to confirm whether their results were ever fetched — so
  deleting via sweep requires both `--delete-sandbox` and `--force`
  together, on top of `--yes`:

  ```bash
  .venv/bin/python orchestrate.py --phase sweep --yes --delete-sandbox --force
  ```

  Run `sweep` after any run you're unsure about, or periodically as a
  general check — it's cheap and doesn't require a state file. Prefer just
  stopping (`--yes` alone) unless you're certain you no longer need a
  sandbox's data.

## Known risk: mount namespace support

The `unshare --mount` and `mount --bind` calls inside the sandbox are what
give each executor an isolated `/submission` directory. If the Daytona
container runtime blocks `unshare` or mount operations, `preflight` will fail
with a clear error. In that case the only remaining options are:

1. Run sequentially with a single executor (so one global `/submission` is
   safe because only one task is active at a time).
2. Use a different host that supports real VMs (e.g. GCP Compute Engine
   with a GPU instance) where Docker runs natively.

## Troubleshooting

**`dsgym: error: unrecognized arguments: --prompt`**

The orchestrator clones `--repo-url` (default `https://github.com/Adamwewe/DSGym.git`)
into the sandbox. If your local `dsgym` repo has a `--prompt` flag that the
GitHub version does not, the sandbox will not support it. Either push your local
changes to GitHub, or omit `--prompt` from the run command.

**`DaytonaConflictError: session already exists`**

This is fixed in the current orchestrator: each `--phase run` now uses a
unique session id. If you see this, pull the latest version of the script.

**`DaytonaNotFoundError: Sandbox ... not found`**

You killed or deleted the sandbox. Run `--phase create` again to provision a
new one, then `bootstrap`, `launch`, and `run`.

**Manager `/allocate` returns `500` with `ValueError: too many values to unpack`**

This is a FastAPI/Pydantic version mismatch. `uv sync` resolves the latest
FastAPI from the lock, but the manager/executor `requirements.txt` files expect
older, pinned versions. The orchestrator now installs the manager into a
separate `.venv-manager` venv so it always has the exact pinned stack.

If you still see this error, the manager is probably using the wrong venv or
an old process is still running. Re-run bootstrap and launch:

```bash
.venv/bin/python orchestrate.py --phase bootstrap
.venv/bin/python orchestrate.py --phase launch
```

**Launch seems to hang**

`launch` is waiting for the manager/executors to become healthy. Use `--phase
exec` to inspect logs:

```bash
.venv/bin/python orchestrate.py --phase exec --command "cat /tmp/dsgym_manager.log"
.venv/bin/python orchestrate.py --phase exec --command "cat /tmp/dsgym_executor_000.log"
.venv/bin/python orchestrate.py --phase exec --command "curl -s http://localhost:5000/health"
```
