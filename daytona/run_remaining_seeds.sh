#!/usr/bin/env bash
# Resume seed-42 (remaining prompts), then run seeds 50 and 60.
# Sandbox from seed-42 is still alive (fetch failed before kill).

set -euo pipefail
cd "$(dirname "$0")"

set -a
source .credentials.env
set +a

export PYTHONUNBUFFERED=1

wait_for() {
  local pattern="$1"
  echo "⏳ Waiting for process matching '$pattern' to finish..."
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep 30
  done
  echo "✅ Done waiting."
}

# If a seed-42 run is still active, wait for it.
if pgrep -f "run_experiments.sh --skip-create" >/dev/null 2>&1; then
  wait_for "run_experiments.sh --skip-create"
fi

run_prompts() {
  local seed="$1"
  shift
  export SEED="$seed"
  echo "SEED: $SEED"

  local DATASET="dspredict-hard"
  local MODEL="gpt-4o-mini"
  local BACKEND="litellm"
  local MAX_WORKERS=4
  local LIMIT=100
  local DAGS="--dags true"

  for entry in "$@"; do
    local prompt="${entry%%:*}"
    local label="${entry##*:}"
    local tag="${label}-seed${SEED}"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "EXPERIMENT: $tag (prompt=$prompt, seed=$SEED)"
    echo "══════════════════════════════════════════════════════════════"

    .venv/bin/python orchestrate.py --phase exec --command "curl -s -X POST http://localhost:5000/reset"
    .venv/bin/python orchestrate.py --phase run \
      --dataset "$DATASET" --model "$MODEL" --backend "$BACKEND" \
      --max-workers "$MAX_WORKERS" --limit "$LIMIT" $DAGS \
      --seed "$SEED" --prompt "$prompt" --tag "$tag"
    .venv/bin/python orchestrate.py --phase fetch
    echo "✅ Experiment '$tag' complete. Results fetched."
  done
}

echo "🔄 Starting background auto-fetch..."
.venv/bin/python orchestrate.py --phase auto-fetch --auto-fetch-interval 1200 &
AUTO_FETCH_PID=$!
trap "kill $AUTO_FETCH_PID 2>/dev/null || true" EXIT

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Resuming seed 42 (remaining 2 prompts: data-quality-methods, expert-agentic)"
echo "══════════════════════════════════════════════════════════════"
echo "📦 Fetching existing results first..."
.venv/bin/python orchestrate.py --phase fetch || echo "⚠️  Fetch failed — continuing anyway"
run_prompts 42 \
  "data-quality-methods:data-quality-methods" \
  "expert-agentic:expert-agentic"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Starting seed 50"
echo "══════════════════════════════════════════════════════════════"
SEED=50 ./run_experiments.sh --skip-create --no-kill 2>&1 | tee run_seed50.log

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Starting seed 60"
echo "══════════════════════════════════════════════════════════════"
SEED=60 ./run_experiments.sh --skip-create 2>&1 | tee run_seed60.log

kill $AUTO_FETCH_PID 2>/dev/null || true
echo ""
echo "🎉 All seeds (42 resumed, 50, 60) complete."
