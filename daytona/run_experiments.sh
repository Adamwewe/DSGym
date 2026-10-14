#!/usr/bin/env bash
# temp.sh — end-to-end DSGym experiment runner.
#
# Usage:
#   ./run_experiments.sh                    # full pipeline: create → bootstrap → launch → experiments → kill
#   ./run_experiments.sh --skip-create      # skip create/bootstrap/launch (reuse existing sandbox)
#   MODEL=claude-sonnet-4-5 SEEDS="40 41 42" ./run_experiments.sh --no-kill  # custom model + seeds, keep sandbox alive
#
# Prerequisites (export before running):
#   export DAYTONA_API_KEY=...
#   export OPENAI_API_KEY=...                         # BACKEND=litellm
#   export AI_CODING_GATEWAY_API_KEY=...              # BACKEND=gateway
#   export AI_CODING_GATEWAY_PROJECT=...              # BACKEND=gateway
#   export KAGGLE_TOKEN=KGAT_...
#   BACKEND=gateway MODEL=claude-opus-4-8 ./run_experiments.sh
#
# The script runs multiple prompt variants sequentially, resetting the
# executor pool between each run to avoid stuck containers.
# A background auto-fetch loop runs throughout, saving results every 20 min
# so they survive even if the laptop dies.

set -euo pipefail
cd "$(dirname "$0")"

SKIP_CREATE=false
NO_KILL=false
BACKEND="${BACKEND:-litellm}"
if [ -z "${MODEL:-}" ]; then
  if [ "$BACKEND" = "gateway" ]; then
    MODEL="claude-opus-4-8"
  else
    MODEL="gpt-4o-mini"
  fi
fi
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
  .venv/bin/python orchestrate.py --phase create --backend "$BACKEND" --cpu 16 --memory 64 --gpu-count 2 --gpu-type RTX_4090
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
DATASET="${DATASET:-dspredict-hard}"
MAX_WORKERS="${MAX_WORKERS:-4}"
LIMIT="${LIMIT:-100}"
# Seeds to cycle through; overridable from outside via SEEDS (space-separated).
SEEDS="${SEEDS:-42 52 53}"
echo "MODEL: $MODEL"
echo "SEEDS: $SEEDS"

run_experiment() {
  local prompt="$1"
  local label="$2"
  local dags="${3:-true}"
  local seed="$4"
  local tag="${label}-seed${seed}"

  # When dags=false, run without a prompt variant and without DAG guidance:
  # dsgym eval must be called with --dags false and no --prompt arg.
  local dags_arg=""
  local prompt_arg=""
  if [ "$dags" = "true" ]; then
    dags_arg="--dags true"
    prompt_arg="--prompt $prompt"
  else
    dags_arg="--dags false"
    prompt_arg=""
  fi

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "EXPERIMENT: $tag (model=$MODEL, prompt=${prompt}, dags=$dags, seed=$seed)"
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
    $dags_arg \
    --seed "$seed" \
    $prompt_arg \
    --tag "${tag}_${seed}"

  # Fetch results immediately after each run.
  .venv/bin/python orchestrate.py --phase fetch

  echo "✅ Experiment '$tag' complete. Results fetched."
}

# Experiment specs: "prompt:label:dags". Run each prompt variant (dags=true)
# plus a non-DAG baseline (dags=false), cycled across all seeds.
EXPERIMENTS=(
  "reasoning-only:nodags:false"
  "reasoning-only:reasoning-only:true"
  "adaptive-revision:adaptive-revision:true"
  "adaptive-revision:adaptive-revision:true"
  "problem-decomposition:problem-decomposition:true"
  "data-quality-methods:data-quality-methods:true"
  "expert-agentic:expert-agentic:true"
)

for seed in $SEEDS; do
  for spec in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r prompt label dags <<< "$spec"
    run_experiment "$prompt" "$label" "$dags" "$seed"
  done
done

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
echo "🎉 All experiments complete (seeds=$SEEDS, model=$MODEL). Results are in daytona/results/."
