#!/usr/bin/env bash
# temp.sh — end-to-end DSGym experiment runner.
#
# Usage:
#   ./run_experiments.sh                    # full pipeline: create → bootstrap → launch → experiments → kill
#   ./run_experiments.sh --skip-create      # skip create/bootstrap/launch (reuse existing sandbox)
#   SEED=50 ./run_experiments.sh --no-kill  # run with a different seed, keep sandbox alive afterwards
#
# Prerequisites (export before running):
#   export DAYTONA_API_KEY=...
#   export OPENAI_API_KEY=...
#   export KAGGLE_TOKEN=KGAT_...
#
# The script runs multiple prompt variants sequentially, resetting the
# executor pool between each run to avoid stuck containers.
# A background auto-fetch loop runs throughout, saving results every 20 min
# so they survive even if the laptop dies.

set -euo pipefail
cd "$(dirname "$0")"

SKIP_CREATE=false
NO_KILL=false
for arg in "$@"; do
  case $arg in
    --skip-create) SKIP_CREATE=true ;;
    --no-kill) NO_KILL=true ;;
  esac
done

# ─── Phase 1: Provision + setup (skip if --skip-create) ────────────────────
if [ "$SKIP_CREATE" = false ]; then
  echo "══════════════════════════════════════════════════════════════"
  echo "PHASE: create + preflight + bootstrap + launch"
  echo "══════════════════════════════════════════════════════════════"
  .venv/bin/python orchestrate.py --phase create --cpu 16 --memory 64 --gpu-count 2 --gpu-type RTX_4090
  .venv/bin/python orchestrate.py --phase preflight
  .venv/bin/python orchestrate.py --phase bootstrap --dataset dspredict-hard
  .venv/bin/python orchestrate.py --phase launch --num-executors 4 --execution-timeout 7200
fi

# ─── Background auto-fetch (safety net) ────────────────────────────────────
# Runs in the background, fetching results every 20 min.
# Killed at the end of the script.
echo "🔄 Starting background auto-fetch (every 20 min)..."
.venv/bin/python orchestrate.py --phase auto-fetch --auto-fetch-interval 1200 &
AUTO_FETCH_PID=$!
echo "   Auto-fetch PID: $AUTO_FETCH_PID"

# Kill auto-fetch on exit (even on error)
trap "kill $AUTO_FETCH_PID 2>/dev/null || true" EXIT

# ─── Phase 2: Run experiments ──────────────────────────────────────────────
DATASET="dspredict-hard"
MODEL="gpt-4o-mini"
BACKEND="litellm"
MAX_WORKERS=4
LIMIT=100
DAGS="--dags true"
SEED="${SEED:-42}"
echo "SEED: $SEED"
run_experiment() {
  local prompt="$1"
  local label="$2"
  local tag="${label}-seed${SEED}"

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "EXPERIMENT: $tag (prompt=$prompt, seed=$SEED)"
  echo "══════════════════════════════════════════════════════════════"

  # Reset the executor pool to clear any stuck containers from the previous run.
  .venv/bin/python orchestrate.py --phase exec --command "curl -s -X POST http://localhost:5000/reset"

  # Run the eval (this blocks until it finishes).
  # The run phase also has its own background auto-fetch thread.
  .venv/bin/python orchestrate.py --phase run \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --max-workers "$MAX_WORKERS" \
    --limit "$LIMIT" \
    $DAGS \
    --seed "$SEED" \
    --prompt "$prompt" \
    --tag "$tag"

  # Fetch results immediately after each run.
  .venv/bin/python orchestrate.py --phase fetch

  echo "✅ Experiment '$tag' complete. Results fetched."
}

# Run each prompt variant.
run_experiment "adaptive-revision"   "adaptive-revision"
run_experiment "problem-decomposition" "problem-decomposition"
run_experiment "data-quality-methods" "data-quality-methods"
run_experiment "expert-agentic"       "expert-agentic"

# ─── Phase 3: Cleanup ──────────────────────────────────────────────────────
# Stop the background auto-fetch first.
kill $AUTO_FETCH_PID 2>/dev/null || true

if [ "$NO_KILL" = true ]; then
  echo ""
  echo "⏭️  Skipping kill (--no-kill). Sandbox left running for the next seed run."
else
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "PHASE: kill (stop billing, auto-fetch first)"
  echo "══════════════════════════════════════════════════════════════"
  .venv/bin/python orchestrate.py --phase kill
fi

echo ""
echo "🎉 All experiments complete (seed=$SEED). Results are in daytona/results/."
