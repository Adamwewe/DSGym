from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class StrictHarnessPolicy:
    require_step_labels: bool = True
    prohibit_step_skipping: bool = True
    require_revision_before_jump: bool = True
    prohibit_step_regression: bool = True
    prohibit_looping: bool = True
    max_identical_message_repeats: int = 1
    max_identical_code_repeats: int = 1
    detect_max_turn_stall: bool = True


@dataclass
class HarnessSampleReport:
    compliance_score: float
    violations: Dict[str, bool]
    counters: Dict[str, float]


class StrictHarnessEvaluator:
    """Strict evaluator for DAG-guided harness compliance.

    Rules are evaluated from the assistant trajectory emitted by agents.
    The implementation is dataset-agnostic so it can be reused in other
    agentic contexts that follow step-labeled planning.
    """

    STEP_PATTERN = re.compile(r"\[Step\s+(\d+)(?:\s*/\s*\d+)?", re.IGNORECASE)
    REVISION_PATTERN = re.compile(r"\[\s*PLAN\s+REVISION", re.IGNORECASE)

    def __init__(self, policy: StrictHarnessPolicy | None = None):
        self.policy = policy or StrictHarnessPolicy()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _extract_python_blocks(text: str) -> List[str]:
        if not text:
            return []
        return [
            block.strip()
            for block in re.findall(r"<python>([\s\S]*?)</python>", text, flags=re.IGNORECASE)
            if block.strip()
        ]

    def _assistant_messages(self, result: Any) -> List[str]:
        trajectory = getattr(result, "trajectory", None) or []
        assistant_messages: List[str] = []

        if isinstance(trajectory, list):
            for step in trajectory:
                if not isinstance(step, dict):
                    continue
                if str(step.get("role", "")).strip().lower() == "assistant":
                    content = str(step.get("content", ""))
                    if content.strip():
                        assistant_messages.append(content)

        if not assistant_messages:
            raw = str(getattr(result, "raw_response", "") or "")
            if raw.strip():
                assistant_messages.append(raw)

        return assistant_messages

    def evaluate_result(self, result: Any, max_turns: int) -> HarnessSampleReport:
        assistant_messages = self._assistant_messages(result)
        step_sequence: List[int] = []
        revision_flags: List[bool] = []

        for message in assistant_messages:
            step_match = self.STEP_PATTERN.search(message)
            if step_match:
                try:
                    step_sequence.append(int(step_match.group(1)))
                except Exception:
                    pass
            revision_flags.append(bool(self.REVISION_PATTERN.search(message)))

        missing_step_labels = bool(
            self.policy.require_step_labels
            and getattr(result, "total_turns", 0) > 1
            and len(step_sequence) == 0
        )

        skipped_step = False
        skipped_step_without_revision = False
        step_regression = False

        if len(step_sequence) >= 2:
            for idx, (prev_step, curr_step) in enumerate(zip(step_sequence, step_sequence[1:]), start=1):
                if curr_step > prev_step + 1 and self.policy.prohibit_step_skipping:
                    skipped_step = True
                    if self.policy.require_revision_before_jump:
                        # Current message index is aligned to idx in assistant_messages.
                        revision_present = revision_flags[idx] if idx < len(revision_flags) else False
                        if not revision_present:
                            skipped_step_without_revision = True
                if curr_step < prev_step and self.policy.prohibit_step_regression:
                    step_regression = True

        normalized_messages = [self._normalize(text) for text in assistant_messages if text.strip()]
        repeated_message_pairs = 0
        for idx in range(1, len(normalized_messages)):
            if normalized_messages[idx] and normalized_messages[idx] == normalized_messages[idx - 1]:
                repeated_message_pairs += 1

        python_blocks: List[str] = []
        for message in assistant_messages:
            python_blocks.extend(self._extract_python_blocks(message))
        normalized_code = [self._normalize(block) for block in python_blocks if block.strip()]

        repeated_code_pairs = 0
        for idx in range(1, len(normalized_code)):
            if normalized_code[idx] and normalized_code[idx] == normalized_code[idx - 1]:
                repeated_code_pairs += 1

        looping = bool(
            self.policy.prohibit_looping
            and (
                repeated_message_pairs > self.policy.max_identical_message_repeats
                or repeated_code_pairs > self.policy.max_identical_code_repeats
            )
        )

        max_turn_stall = bool(
            self.policy.detect_max_turn_stall
            and getattr(result, "total_turns", 0) >= max_turns
            and not getattr(result, "success", False)
        )

        violations = {
            "missing_step_labels": missing_step_labels,
            "skipped_step": skipped_step,
            "skipped_step_without_revision": skipped_step_without_revision,
            "step_regression": step_regression,
            "looping": looping,
            "max_turn_stall": max_turn_stall,
        }
        violation_count = sum(1 for hit in violations.values() if hit)
        compliance_score = max(0.0, 1.0 - (violation_count / len(violations)))

        counters = {
            "assistant_messages": float(len(assistant_messages)),
            "labeled_steps": float(len(step_sequence)),
            "repeated_message_pairs": float(repeated_message_pairs),
            "repeated_code_pairs": float(repeated_code_pairs),
        }

        return HarnessSampleReport(
            compliance_score=compliance_score,
            violations=violations,
            counters=counters,
        )

    def summarize(self, results: List[Any], max_turns: int) -> Dict[str, Any]:
        if not results:
            return {
                "policy": self.policy.__dict__,
                "avg_compliance": 0.0,
                "violation_rates": {
                    "missing_step_labels": 0.0,
                    "skipped_step": 0.0,
                    "skipped_step_without_revision": 0.0,
                    "step_regression": 0.0,
                    "looping": 0.0,
                    "max_turn_stall": 0.0,
                },
                "counters": {
                    "assistant_messages_mean": 0.0,
                    "labeled_steps_mean": 0.0,
                    "repeated_message_pairs_mean": 0.0,
                    "repeated_code_pairs_mean": 0.0,
                },
            }

        per_sample = [self.evaluate_result(result, max_turns=max_turns) for result in results]

        keys = [
            "missing_step_labels",
            "skipped_step",
            "skipped_step_without_revision",
            "step_regression",
            "looping",
            "max_turn_stall",
        ]
        violation_rates = {
            key: sum(1 for sample in per_sample if sample.violations.get(key, False)) / len(per_sample)
            for key in keys
        }

        counters = {
            "assistant_messages_mean": statistics.fmean(sample.counters["assistant_messages"] for sample in per_sample),
            "labeled_steps_mean": statistics.fmean(sample.counters["labeled_steps"] for sample in per_sample),
            "repeated_message_pairs_mean": statistics.fmean(
                sample.counters["repeated_message_pairs"] for sample in per_sample
            ),
            "repeated_code_pairs_mean": statistics.fmean(
                sample.counters["repeated_code_pairs"] for sample in per_sample
            ),
        }

        return {
            "policy": self.policy.__dict__,
            "avg_compliance": statistics.fmean(sample.compliance_score for sample in per_sample),
            "violation_rates": violation_rates,
            "counters": counters,
        }
