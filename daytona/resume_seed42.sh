#!/usr/bin/env bash
# Resume seed 42 after local log-stream disconnect (eval may still run in sandbox).
set -euo pipefail
cd "$(dirname "$0")"
set -a && source .credentials.env && set +a
export PYTHONUNBUFFERED=1

eval_running() {
  .venv/bin/python orchestrate.py --phase exec --command "pgrep -f 'dsgym eval'" 2>/dev/null \
    | grep -qE 'root|[0-9]+'
}

echo "⏳ Waiting for in-flight eval to finish..."
while eval_running; do
  sleep 120
  n=$(.venv/bin/python orchestrate.py --phase exec --command "ls /root/DSGym/evaluation_results/*.json 2>/dev/null | wc -l" 2>/dev/null | grep -E '^[0-9]+$' | tail -1 || echo "?")
  echo "   results so far: ${n:-?}"
done
echo "✅ Eval finished."

.venv/bin/python orchestrate.py --phase fetch || true

run_one() {
  local prompt="$1" label="$2"
  local tag="${label}-seed42"
  echo ""; echo "EXPERIMENT: $tag"
  .venv/bin/python orchestrate.py --phase exec --command "curl -s -X POST http://localhost:5000/reset"
  .venv/bin/python orchestrate.py --phase run \
    --dataset dspredict-hard --model gpt-4o-mini --backend litellm \
    --max-workers 4 --limit 100 --dags true --seed 42 \
    --prompt "$prompt" --tag "$tag" || echo "⚠️  run failed for $tag"
  .venv/bin/python orchestrate.py --phase fetch || true
}

for p in problem-decomposition data-quality-methods expert-agentic; do
  run_one "$p" "$p"
done

.venv/bin/python orchestrate.py --phase kill
echo "🎉 Seed 42 complete."
