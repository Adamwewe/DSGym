#!/usr/bin/env bash
# Wait for in-flight eval, then finish seed 42 + run seeds 50/60.
set -euo pipefail
cd "$(dirname "$0")"
set -a && source .credentials.env && set +a
export PYTHONUNBUFFERED=1

eval_running() {
  .venv/bin/python orchestrate.py --phase exec --command "pgrep -f 'dsgym eval'" 2>/dev/null \
    | grep -qE '^[0-9]+$|root'
}

echo "⏳ Waiting for in-flight dsgym eval to finish in sandbox..."
while eval_running; do
  sleep 120
  count=$(.venv/bin/python orchestrate.py --phase exec --command \
    "ls /root/DSGym/evaluation_results/*data-quality-methods* 2>/dev/null | wc -l" 2>/dev/null \
    | grep -E '^[0-9]+$' | tail -1 || echo "?")
  echo "   data-quality-methods results so far: ${count:-?}"
done
echo "✅ Eval finished in sandbox."

.venv/bin/python orchestrate.py --phase fetch || echo "⚠️  Fetch failed — continuing"

run_prompts() {
  local seed="$1"
  shift
  export SEED="$seed"
  local DATASET="dspredict-hard" MODEL="gpt-4o-mini" BACKEND="litellm"
  local MAX_WORKERS=4 LIMIT=100 DAGS="--dags true"
  for entry in "$@"; do
    local prompt="${entry%%:*}"
    local label="${entry##*:}"
    local tag="${label}-seed${SEED}"
    echo ""
    echo "EXPERIMENT: $tag"
    .venv/bin/python orchestrate.py --phase exec --command "curl -s -X POST http://localhost:5000/reset"
    .venv/bin/python orchestrate.py --phase run \
      --dataset "$DATASET" --model "$MODEL" --backend "$BACKEND" \
      --max-workers "$MAX_WORKERS" --limit "$LIMIT" $DAGS \
      --seed "$SEED" --prompt "$prompt" --tag "$tag" || echo "⚠️  Run failed for $tag — continuing"
    .venv/bin/python orchestrate.py --phase fetch || echo "⚠️  Fetch failed for $tag"
  done
}

.venv/bin/python orchestrate.py --phase auto-fetch --auto-fetch-interval 1200 &
AUTO_FETCH_PID=$!
trap "kill $AUTO_FETCH_PID 2>/dev/null || true" EXIT

echo "Starting expert-agentic (seed 42)..."
run_prompts 42 "expert-agentic:expert-agentic"

echo ""
echo "🛑 Seed 42 done — skipping seeds 50/60 (use finish_seed42_shutdown.sh or --phase kill)."
kill $AUTO_FETCH_PID 2>/dev/null || true
echo "🎉 Seed 42 experiments complete."
