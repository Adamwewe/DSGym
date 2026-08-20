#!/usr/bin/env bash
# Wait for seed 42 to finish, fetch results, shut down all sandboxes.
# Does NOT run seeds 50 or 60.
set -euo pipefail
cd "$(dirname "$0")"
set -a && source .credentials.env && set +a
export PYTHONUNBUFFERED=1

# Cancel any pipeline that would continue to seeds 50/60.
pkill -f "wait_and_finish.sh" 2>/dev/null || true
pkill -f "run_remaining_seeds.sh" 2>/dev/null || true

eval_running() {
  .venv/bin/python orchestrate.py --phase exec --command "pgrep -f 'dsgym eval'" 2>/dev/null \
    | grep -qE 'root|[0-9]+'
}

echo "⏳ Waiting for seed 42 eval(s) to finish in sandbox..."
while eval_running; do
  sleep 60
  counts=$(.venv/bin/python orchestrate.py --phase exec --command \
    "cd /root/DSGym/evaluation_results && for t in adaptive-revision problem-decomposition data-quality-methods expert-agentic; do echo \$t:\$(ls *\$t* 2>/dev/null|wc -l); done" 2>/dev/null \
    | grep -E '^[a-z].*:[0-9]+' | tr '\n' ' ' || true)
  echo "   Progress: ${counts:-unknown}"
done
echo "✅ Seed 42 eval finished."

echo "📦 Fetching results..."
.venv/bin/python orchestrate.py --phase fetch || echo "⚠️  Fetch failed — results may still be on volume"

echo "🛑 Stopping sandbox from state file..."
.venv/bin/python orchestrate.py --phase kill || echo "⚠️  Kill via state file failed"

echo "🧹 Sweeping any remaining DSGym sandboxes..."
.venv/bin/python orchestrate.py --phase sweep --yes || true

echo "✅ Seed 42 complete. All sandboxes stopped."
