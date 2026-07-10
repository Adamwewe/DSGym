# Trace Analysis V6 — `sonnet-4-5` / DAGs: Total Harness Failure (May 30 15:19)

> **Scope**: Analysis of the `sonnet-4-5 + dags` run (May 30 15:19) on DSPredict-Sampled
> (35 Kaggle competitions). This run pre-dates the renamed `claude-sonnet-4-5` runs analyzed
> in V4/V5. It produced **zero completions** due to a harness-level configuration failure.

---

## 1 · Quick Summary

| Run | Complete | Error | No-Submit | Avg pub pct |
|-----|----------|-------|-----------|-------------|
| **sonnet-4-5 / dags (this run)** | **0** | **0** | **35** | — |
| Claude-4.5 / dags (V5, Jun 7) | 16 | 5 | 14 | 25.7% |
| Claude-4.5 / dags (V4, old) | 17 | 4 | 14 | 27.0% |
| GPT-5.1 / dags | 25 | 3 | 7 | 24.2% |

**Every single task failed before the agent ran a single turn.** This is not a model-quality
failure — it is a harness crash that prevented code execution entirely.

---

## 2 · What Actually Happened

All 35 tasks share identical failure characteristics:

| Field | Value |
|-------|-------|
| `total_turns` | 0 |
| `execution_time` | ~0.05s |
| `error_info` | `"unsupported format string passed to NoneType.__format__"` |
| `submission_status` | `submission file not found` |
| `score` | `None` |

~0.05s total execution time means the agent never began meaningful work. For comparison,
a single LLM call takes 1–30s. Something crashed in harness initialisation.

---

## 3 · Root Cause Trace

The failure chain involves **two concurrent bugs**, one masking the other:

### Bug A (Primary) — Invalid LiteLLM Model Name

**Config**: `model_name: "sonnet-4-5"`, `backend_type: "litellm"`

In `dsgym/agents/backends/litellm_backend.py`, `_setup_litellm()` selects the API key by
matching the model name prefix:

```python
if self.model_name.startswith("together_ai/"):   → TOGETHER_API_KEY
elif self.model_name.startswith("gpt-") or ...:  → OPENAI_API_KEY
elif self.model_name.startswith("claude-") or self.model_name.startswith("anthropic/"):
                                                  → ANTHROPIC_API_KEY
else:
    litellm.api_key = self.api_key  # = None (no env var matched)
```

`"sonnet-4-5"` does not match any branch. No API key is set. When `litellm.completion(model="sonnet-4-5", ...)` is called:
- LiteLLM does not recognise `"sonnet-4-5"` as a valid provider-qualified model name
- It raises `BadRequestError` or `AuthenticationError` immediately
- `BadRequestError` is explicitly not retried (caught and re-raised at line 154–156 of the backend)
- This failure propagates to the inner `try/except` in `dspredict_react_agent.py` (line 118–127)
- The inner handler adds an error message to conversation and continues to the next turn
- `actual_turns` is never incremented (it is only set AFTER `env.step()` succeeds, line 98)
- All 15 turns fail in rapid succession (~3ms each) → `actual_turns = 0`, total time ~0.05s

**Fix**: Use the correct LiteLLM-qualified model name: `"claude-sonnet-4-5"` (for the
Anthropic API) or `"anthropic/claude-sonnet-4-5-20251202"` (provider-explicit form).
The later runs (`claude_sonnet_4_5` in V4/V5) used the correct name and worked.

### Bug B (Secondary) — Container Allocation Failure Masked by Format Error

`env.init()` in `dsgym/agents/environment/envs/allocated_code/env.py` silently swallows
container allocation failures:

```python
try:
    container_id = self.tool_group.allocate_container()
except Exception as e:
    print(f"⚠️ Warning: Failed to allocate container: {e}")
    # execution continues — allocated_container stays None
```

If the container manager (`http://localhost:5000/allocate`) is unreachable — which is likely
for a run that crashed before containers were set up — `allocated_container` remains `None`.

After all 15 turns fail (Bug A), execution reaches `dspredict_react_agent.py` line 135:

```python
container_dir = os.path.join(self.submission_dir, f"container_{container_id:03d}")
#                                                              ^^^^^^^^^^^^^^^^^^^
#                                           container_id is None → TypeError here
```

The `:03d` format spec cannot be applied to `None`, raising:
> `TypeError: unsupported format string passed to NoneType.__format__`

This is caught by the outer `except Exception as e:` (line 230) and stored as `error_info`.
It **completely obscures Bug A** — the real failure (invalid model name → zero LLM responses)
never appears in the output.

Note: line 74–80 correctly guards the stale-submission cleanup with `if container_id is not None:`,
but line 135 has no such guard.

---

## 4 · Why This Run Pre-Dates the Working Runs

| Config field | This run (`sonnet-4-5`) | Later runs (`claude-sonnet-4-5`) |
|---|---|---|
| `model_name` | `sonnet-4-5` | `claude-sonnet-4-5` |
| `timestamp` | 2026-05-30T15:19:34 | 2026-06-07 (V5) |
| `max_tokens` | 1524 | (not shown, but much higher) |
| `max_workers` | 24 | varies |

The `sonnet-4-5` name appears to be an early naming attempt before the full
`claude-sonnet-4-5` convention was established. All subsequent runs use the
provider-prefixed name and succeed.

---

## 5 · No Competition-Level Failures to Analyse

Because the harness crashed before running any agent turns, there is no per-competition
behaviour to examine. The failure pattern is identical across all 35 tasks:

- 0 LLM calls made
- 0 code cells executed
- No submission file generated
- Same 0.05s crash time

This is fundamentally different from the NO-SUBMIT cases in V4/V5 where the agent
actually attempted the task but ran out of turns or hit ImportErrors.

---

## 6 · Bugs to Fix

### 6.1 Missing Guard on `container_id` at Line 135 (Code Bug)

**File**: `dsgym/agents/dspredict_react_agent.py`, line 135

**Problem**: `container_id` can be `None` if container allocation failed (caught silently in
`env.init()`). The subsequent format string `f"container_{container_id:03d}"` raises
`TypeError`, masking the real failure.

**Fix**:
```python
# Before (line 135):
container_dir = os.path.join(self.submission_dir, f"container_{container_id:03d}")

# After:
if container_id is None:
    print("⚠️  No container allocated — cannot locate submission file")
    return {
        "solution": "",
        "success": False,
        "turns": actual_turns,
        "error": "Container allocation failed; no submission file available",
        ...
    }
container_dir = os.path.join(self.submission_dir, f"container_{container_id:03d}")
```

### 6.2 Model Name Validation (Configuration / UX Bug)

**File**: `dsgym/agents/backends/litellm_backend.py`, `_setup_litellm()`

**Problem**: Unknown model names fall through the API key detection silently, producing
`AuthenticationError` at call time rather than a clear startup failure.

**Fix**: Add a warning or validation when no provider is detected:
```python
else:
    # Unknown provider — try whatever is in the environment or api_key arg
    if self.api_key:
        litellm.api_key = self.api_key
    else:
        print(f"⚠️  WARNING: Model '{self.model_name}' does not match a known provider "
              f"prefix (claude-, gpt-, anthropic/, together_ai/). "
              f"Ensure ANTHROPIC_API_KEY/OPENAI_API_KEY/etc is set, or use a "
              f"provider-qualified name like 'claude-sonnet-4-5'.")
```

### 6.3 Container Allocation Failure Should Propagate (Design Bug)

**File**: `dsgym/agents/environment/envs/allocated_code/env.py`, `init()` lines 111–115

**Problem**: Container allocation failure is silently swallowed. The agent then proceeds to
run turns that all fail (no container to execute code in), wasting the turn budget and
producing misleading output.

**Fix**: Re-raise after logging, or set a `self.allocation_failed` flag that short-circuits
`env.step()` immediately with a clear error rather than confusing downstream code.

---

## 7 · Recommendations

| Priority | Change | Impact |
|----------|--------|--------|
| 🔴 HIGH | Guard `container_id` before format string at line 135 of `dspredict_react_agent.py` | Prevents misleading `NoneType.__format__` errors masking real failures |
| 🔴 HIGH | Always use provider-prefixed model names (`claude-sonnet-4-5`, not `sonnet-4-5`) | Prevents silent API key misconfiguration |
| 🟡 MED | Add model name validation / warning in `_setup_litellm()` | Surfaces misconfiguration at agent init time, not mid-run |
| 🟡 MED | Propagate container allocation failure instead of swallowing it | Prevents 15 useless turn attempts when infrastructure is down |
| 🟢 LOW | Log the actual `generate()` error inside the inner turn except handler | Would surface "AuthenticationError: invalid model sonnet-4-5" in traces |
