"""
Code execution-based evaluation metrics.
"""

import re
import ast
import subprocess
import tempfile
import resource
import os
from typing import Optional, Dict, Any
from .base import BaseMetric, MetricResult


class CodeExecutionMetric(BaseMetric):
    """
    Metric that evaluates code execution correctness.
    """
    
    def __init__(self, timeout: int = 30, **kwargs):
        """
        Initialize code execution metric.
        
        Args:
            timeout: Maximum execution time in seconds
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "code_execution"
    
    @property
    def requires_ground_truth(self) -> bool:
        return False  # Can evaluate code independently
    
    def _extract_code(self, text: str) -> Optional[str]:
        """
        Extract Python code from text.
        """
        if not text:
            return None
        
        # Look for code blocks
        patterns = [
            r'```python\s*(.*?)\s*```',  # Python code blocks
            r'```\s*(.*?)\s*```',        # Generic code blocks
            r'<python>\s*(.*?)\s*</python>',  # Custom python tags
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code:
                    return code
        
        # If no code blocks found, check if the entire text looks like code
        try:
            ast.parse(text)
            return text
        except SyntaxError:
            pass
        
        return None
    
    def _validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate code before execution.

        Returns (ok, reason).  Only rejects code that is definitively
        unsound (oversized, unparseable).  Blocklist-based "safety" checks
        are NOT used here because they are trivially bypassed via attribute
        access, importlib, or __builtins__; real sandboxing is provided by
        the per-process resource limits applied in _execute_code.
        """
        if not code:
            return False, "Empty code"
        # Reject unreasonably large payloads (>512 KB) to prevent memory abuse.
        if len(code) > 512 * 1024:
            return False, f"Code too large ({len(code)} bytes; limit 524288)"
        # Reject syntactically invalid code early.
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        return True, ""
    
    def _execute_code(self, code: str) -> dict:
        """
        Execute Python code in a subprocess with hard resource limits and
        return results.
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        def _set_resource_limits():
            """Applied in the child process before exec."""
            # 512 MB virtual memory ceiling
            mem = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            # No new child processes from the evaluated code
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

        try:
            import time
            start_time = time.time()

            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=_set_resource_limits,
            )

            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None,
                "output": result.stdout,
                "return_code": result.returncode,
                "execution_time": execution_time
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Code execution timed out after {self.timeout} seconds",
                "output": "",
                "execution_time": self.timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0.0
            }
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate code execution.
        
        Args:
            prediction: Model prediction containing code
            ground_truth: Ground truth (unused for basic execution)
            query: Original query (unused)
            **kwargs: Additional context
            
        Returns:
            MetricResult with execution score
        """
        # Extract code from prediction
        code = self._extract_code(prediction)

        if code is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={
                    "reason": "No executable code found in prediction",
                    "prediction": prediction
                }
            )

        ok, reason = self._validate_code(code)
        if not ok:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"reason": reason, "prediction": prediction}
            )

        # Execute code
        exec_result = self._execute_code(code)
        
        # Score based on successful execution
        score = 1.0 if exec_result["success"] else 0.0
        
        details = {
            "extracted_code": code,
            "execution_success": exec_result["success"],
            "execution_time": exec_result["execution_time"],
            "output": exec_result["output"],
        }
        
        if not exec_result["success"]:
            details["error"] = exec_result["error"]
            details["return_code"] = exec_result.get("return_code")
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )


class CodeCorrectnessMetric(CodeExecutionMetric):
    """
    Metric that evaluates code correctness by comparing execution output.
    """
    
    @property
    def name(self) -> str:
        return "code_correctness"
    
    @property
    def requires_ground_truth(self) -> bool:
        return True  # Need expected output for comparison
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate code correctness by comparing output.
        
        Args:
            prediction: Model prediction containing code
            ground_truth: Expected output or correct code
            query: Original query (unused)
            **kwargs: Additional context
            
        Returns:
            MetricResult with correctness score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        # Extract code from prediction
        code = self._extract_code(prediction)
        
        if code is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={
                    "reason": "No executable code found in prediction",
                    "prediction": prediction
                }
            )
        
        # Execute code
        exec_result = self._execute_code(code)
        
        if not exec_result["success"]:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={
                    "reason": "Code execution failed",
                    "extracted_code": code,
                    "execution_error": exec_result["error"],
                    "execution_time": exec_result["execution_time"]
                }
            )
        
        # Compare output with ground truth
        actual_output = exec_result["output"].strip()
        expected_output = ground_truth.strip()
        
        # Simple string comparison (can be enhanced)
        is_correct = actual_output == expected_output
        score = 1.0 if is_correct else 0.0
        
        details = {
            "extracted_code": code,
            "execution_success": True,
            "execution_time": exec_result["execution_time"],
            "actual_output": actual_output,
            "expected_output": expected_output,
            "output_match": is_correct,
        }
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )