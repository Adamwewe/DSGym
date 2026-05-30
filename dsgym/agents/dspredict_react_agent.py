"""
DSPredictReActAgent - specialized ReAct agent for DSPredict-style competitions.

Extends ReActDSAgent to explicitly capture and return detailed per-turn
trajectories suitable for downstream analysis of DSPredict runs.
"""

import time
import traceback
from typing import Dict, Any, List
import os
from datetime import datetime
import shutil

from .react_ds_agent import ReActDSAgent
from .environment import AllocatedCodeEnv


class DSPredictReActAgent(ReActDSAgent):
    """ReAct agent tuned for DSPredict challenges, returning rich trajectories."""
    def __init__(self, backend: str, model: str, **kwargs):
        super().__init__(backend, model, **kwargs)

        self.submission_dir = kwargs.get('submission_dir', '')
        # Optional HTTP client timeout (seconds) for code execution manager
        self.time_out = kwargs.get('time_out', 1800)

    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve a DSPredict task and return detailed trajectories.

        Returns a dict with keys including:
        - solution: final answer string (may include <answer> tags content)
        - success: bool
        - turns: number of turns executed
        - error: optional error message
        - metadata: run metadata (model, backend, timings, etc.)
        - conversation: full conversation history
        - trajectory: List[Dict] with per-turn details
        - raw_result: compact summary
        """
        start_time = time.time()

        trajectory: List[Dict[str, Any]] = []

        try:
            # Extract conversation and metadata from sample
            conversation = sample.get("prompt", [])
            if not conversation:
                raise ValueError("Sample must contain 'prompt' field with conversation")

            extras = {
                "reward_spec": sample.get("reward_spec", {"ground_truth": ""}),
                "extra_info": sample.get("extra_info", {}),
                "max_turns": self.max_turns,
            }

            # Create environment for this task (local to this call)
            env = AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_turns,
                output_dir=self.output_dir,
                time_out=self.time_out,
                container_type=self.container_type,
            )

            # Initialize environment
            conversation, _ = env.init(conversation, **extras)
            container_id = env.tool_group.allocated_container

            # Clear any stale submission.csv left by a previous competition that ran
            # on this container.  The /submission directory is a host-mounted volume
            # so the file survives the kernel restart that happens on deallocate.
            if container_id is not None:
                stale_submission = os.path.join(
                    self.submission_dir, f"container_{container_id:03d}", "submission.csv"
                )
                if os.path.exists(stale_submission):
                    os.remove(stale_submission)
                    print(f"🧹 Cleared stale submission from previous competition: {stale_submission}")
            total_tokens = 0
            final_answer = ""
            actual_turns = 0

            for turn in range(self.max_turns):
                try:
                    # Generate response (prune history first to stay within context)
                    pruned = self._prune_conversation(conversation)
                    response = self.backend_instance.generate(pruned)

                    # Step environment
                    step_start = time.time()
                    step_output = env.step(response)
                    step_time = time.time() - step_start

                    # Count tokens (approximate)
                    total_tokens += len(response.split())
                    actual_turns = turn + 1

                    # Update conversation with assistant response and new observations
                    conversation.append({
                        "role": "assistant",
                        "content": step_output.get("postprocessed_action", response),
                    })
                    trajectory = self.append_traj(trajectory, turn, "assistant", response, step_output.get("done", False), step_output.get("reward", 0.0), step_time)

                    if step_output["observations"]:
                        conversation.extend(step_output["observations"])
                        trajectory = self.append_traj(trajectory, turn, "user", step_output["observations"][0]["content"], step_output.get("done", False), step_output.get("reward", 0.0), step_time)
                    else:
                        # no observation means that there is no new code produced, we are reaching the end. 
                        step_output["done"] = True
                    # Check if task is complete
                    if step_output["done"]:
                        final_answer = step_output["metadata"].get("final_answer", response)
                        break

                except Exception as step_err:
                    error_msg = f"Turn {turn + 1} failed: {step_err}"
                    # Add error to conversation for recovery
                    conversation.append({
                        "role": "user",
                        "content": f"Error: {error_msg}. Please try a different approach.",
                    })
                    # Record error step in trajectory
                    trajectory = self.append_traj(trajectory, turn, "user", error_msg, False, 0.0, 0.0)
                    continue

            # Save prediction if we have an answer
            if final_answer:
                prefix = sample.get("extra_info", {}).get("id", "temp")
                env.save_prediction(final_answer, filename_prefix=prefix)

            execution_time = time.time() - start_time
            container_dir = os.path.join(self.submission_dir, f"container_{container_id:03d}")
            submission_file = os.path.join(container_dir, "submission.csv")
            submission_path = ""
            print(f"{submission_file=}")
            success = False
            if os.path.exists(submission_file):
                challenge_name = sample.get("extra_info", {}).get("challenge_name", "")

                # ── submission format validation ──────────────────────────────
                # Check the generated CSV against sample_submission.csv on the
                # host.  If format errors are found, inject them back into the
                # conversation and give the agent up to 5 extra turns to fix
                # the file before we accept it.
                try:
                    from dsgym.eval.metrics.dspredict.dspredict_metric import _validate_submission
                    MAX_FIX_TURNS = 5
                    for fix_attempt in range(MAX_FIX_TURNS + 1):
                        errors = _validate_submission(submission_file, challenge_name)
                        if not errors:
                            break  # submission is valid
                        if fix_attempt == MAX_FIX_TURNS:
                            print(f"⚠️  Submission still invalid after {MAX_FIX_TURNS} fix turns — uploading anyway")
                            break

                        error_lines = "\n".join(f"  - {e}" for e in errors)
                        print(f"⚠️  submission.csv format errors (fix attempt {fix_attempt + 1}):\n{error_lines}")
                        fix_msg = (
                            f"Your submission.csv has format errors that Kaggle will reject:\n{error_lines}\n\n"
                            f"The expected format is defined in "
                            f"/data/dspredict-sampled/{challenge_name}/sample_submission.csv inside the container.\n"
                            f"Fix the column names and/or row count, overwrite submission.csv, and confirm it matches "
                            f"the sample exactly (same columns, same number of rows)."
                        )
                        conversation.append({"role": "user", "content": fix_msg})

                        # Run one more agent turn to fix the file
                        try:
                            pruned = self._prune_conversation(conversation)
                            response = self.backend_instance.generate(pruned)
                            step_output = env.step(response)
                            actual_turns += 1
                            conversation.append({
                                "role": "assistant",
                                "content": step_output.get("postprocessed_action", response),
                            })
                            trajectory = self.append_traj(
                                trajectory, actual_turns, "assistant", response,
                                step_output.get("done", False), step_output.get("reward", 0.0), 0.0
                            )
                            if step_output.get("observations"):
                                conversation.extend(step_output["observations"])
                                trajectory = self.append_traj(
                                    trajectory, actual_turns, "user",
                                    step_output["observations"][0]["content"],
                                    step_output.get("done", False), step_output.get("reward", 0.0), 0.0
                                )
                        except Exception as fix_err:
                            print(f"Fix turn {fix_attempt + 1} execution error: {fix_err}")
                            break
                except Exception as val_err:
                    print(f"⚠️  Could not run submission validation: {val_err}")
                # ─────────────────────────────────────────────────────────────

                success = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_name = f"{challenge_name}_{container_id}_{timestamp}_submission.csv"
                unique_path = os.path.join(container_dir, unique_name)
                shutil.copy2(submission_file, unique_path)
                print(f"💾 Submission file saved locally: {unique_path}")
                submission_path = unique_path


            return {
                "solution": submission_path,
                "success": success,
                "turns": actual_turns,
                "error": None,
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "max_turns": self.max_turns,
                    "total_tokens": total_tokens,
                    "execution_time": execution_time,
                    "conversation_length": len(conversation),
                },
                "conversation": conversation,
                "trajectory": trajectory,
                "raw_result": {
                    "prediction": submission_path,
                    "turns": actual_turns,
                    "total_tokens": total_tokens,
                },
            }

        except Exception as e:
            
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            print(F"Error in agent: {error_trace}")

            return {
                "solution": "",
                "success": False,
                "turns": 0,
                "error": str(e),
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "max_turns": self.max_turns,
                    "execution_time": execution_time,
                    "error_trace": error_trace,
                },
                "conversation": [],
                "trajectory": trajectory,
                "raw_result": None,
            }
        finally:
            if "env" in locals():
                env.close()

    def append_traj(self, trajectory, turn, role, content, done, reward, step_time):
        # Record trajectory step
        trajectory.append({
            "turn": turn,
            "role": role,
            "content": content,
            "done": done,
            "reward": reward,
            "step_time": step_time,
        })
        return trajectory

    def _prune_conversation(self, conversation: List[Dict[str, Any]], keep_recent: int = 20) -> List[Dict[str, Any]]:
        """
        Trim the conversation to avoid context-window overflow.

        Always keeps:
          - messages[0]: system prompt
          - messages[1]: initial user task description
          - The most recent ``keep_recent`` messages

        Middle turns (verbose code outputs from early steps) are dropped with a
        placeholder so the model knows history was truncated.
        """
        if len(conversation) <= keep_recent + 2:
            return conversation

        head = conversation[:2]
        tail = conversation[-keep_recent:]
        dropped = len(conversation) - 2 - keep_recent
        placeholder = {
            "role": "user",
            "content": (
                f"[{dropped} earlier turn(s) omitted to stay within the context window. "
                "Variables and files produced in those turns are still available in the execution environment.]"
            ),
        }
        return head + [placeholder] + tail
