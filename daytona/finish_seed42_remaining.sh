#!/usr/bin/env bash
# Finish seed 42: wait for in-flight eval, run expert-agentic, fetch, kill.
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

echo ""
echo "EXPERIMENT: expert-agentic-seed42"
.venv/bin/python orchestrate.py --phase exec --command "curl -s -X POST http://localhost:5000/reset"
.venv/bin/python orchestrate.py --phase run \
  --dataset dspredict-hard --model gpt-4o-mini --backend litellm \
  --max-workers 4 --limit 100 --dags true --seed 42 \
  --prompt expert-agentic --tag expert-agentic-seed42 \
  || echo "⚠️  run failed (eval may still be running in sandbox)"

while eval_running; do
  sleep 120
  n=$(.venv/bin/python orchestrate.py --phase exec --command "ls /root/DSGym/evaluation_results/*.json 2>/dev/null | wc -l" 2>/dev/null | grep -E '^[0-9]+$' | tail -1 || echo "?")
  echo "   results so far: ${n:-?}"
done

.venv/bin/python orchestrate.py --phase fetch || true
.venv/bin/python orchestrate.py --phase kill
echo "🎉 Seed 42 complete."
