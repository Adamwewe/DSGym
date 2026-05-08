"""
DSPredict submission metric for DSGym.

This metric submits a generated `submission.csv` to a DSPredict competition,
waits for the score, and augments it with leaderboard statistics. It mirrors
the logic from the legacy evaluator's `evaluate_dspredict_submission_in_leaderboard`.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from datetime import datetime
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random, wait_exponential_jitter, retry_if_exception

from ..base import BaseMetric, MetricResult


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that should trigger a retry."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("429", "too many requests", "rate limit", "timeout", "connection aborted",
                   "broken pipe", "reset by peer", "temporarily unavailable", "503", "502", "500")
    )


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _progress_reader(file_path: str, file_size: int, chunk_size: int = 1024 * 1024):
    """Generator that streams a file in chunks while printing upload progress.

    Streams directly from disk so we never load the full file into RAM — important
    for large submission files (hundreds of MB).  httpx accepts a generator as
    ``content`` when ``Content-Length`` is also provided.
    """
    uploaded = 0
    last_report = 0
    report_every = max(file_size // 20, chunk_size)  # report ~every 5%
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            uploaded += len(chunk)
            yield chunk
            if uploaded - last_report >= report_every or uploaded == file_size:
                pct = 100.0 * uploaded / file_size if file_size else 100
                print(f"  … {uploaded / (1024*1024):.1f} / {file_size / (1024*1024):.1f} MB  ({pct:.0f}%)",
                      flush=True)
                last_report = uploaded


class DSPredictSubmissionManager:
    """Lightweight DSPredict submission manager using KaggleSDK (already installed)."""

    def __init__(self, username: Optional[str] = None):
        try:
            import json
            from kagglesdk import KaggleClient
            from dsgym.eval.metrics.dspredict.leaderboard_utils import KaggleScraper  # local fallback if present

            # Prefer repo-local kaggle.json, fallback to default path
            creds = None
            if os.path.exists("kaggle.json"):
                with open("kaggle.json", "r") as f:
                    creds = json.load(f)
            else:
                default_path = os.path.expanduser("~/.kaggle/kaggle.json")
                if os.path.exists(default_path):
                    with open(default_path, "r") as f:
                        creds = json.load(f)

            if creds:
                os.environ["KAGGLE_USERNAME"] = creds["username"]
                os.environ["KAGGLE_KEY"] = creds["key"]

            self.client = KaggleClient()
            self.username = username or (creds.get("username") if creds else None)
            try:
                self.scraper = KaggleScraper()  # optional utility, not critical
            except Exception:
                self.scraper = None
        except Exception as e:
            print(f"Failed to initialize KaggleSDK client: {e}")
            self.client = None
            self.username = username
            self.scraper = None

    def submit_file(self, competition_name: str, file_path: str, description: str) -> str:
        from kagglesdk.competitions.types.competition_api_service import (
            ApiCreateSubmissionRequest,
            ApiStartSubmissionUploadRequest,
        )
        import httpx

        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_size_mb = file_size / (1024 * 1024)

        upload_request = ApiStartSubmissionUploadRequest()
        upload_request.competition_name = competition_name
        upload_request.content_length = file_size
        upload_request.file_name = file_name

        last_exc = None
        for attempt in range(3):
            if attempt > 0:
                wait_secs = 15 * attempt
                print(f"⚠️  Upload attempt {attempt + 1} for {competition_name}, retrying in {wait_secs}s…")
                time.sleep(wait_secs)

            # Fetch a fresh presigned URL for every attempt – the old one may
            # have been invalidated by a previous failed PUT.
            upload_response = self.client.competitions.competition_api_client.start_submission_upload(upload_request)

            try:
                # Use httpx so we can set write=None (no write-side timeout).
                # requests has no configurable write timeout; large files can
                # stall mid-upload causing TimeoutError on the write operation.
                timeout = httpx.Timeout(connect=30.0, read=600.0, write=None, pool=30.0)
                print(f"⬆️  Uploading {file_name} ({file_size_mb:.1f} MB) to {competition_name}…")
                upload_start = time.time()
                with httpx.Client(timeout=timeout) as client:
                    response = client.put(
                        upload_response.create_url,
                        content=_progress_reader(file_path, file_size),
                        headers={"Content-Length": str(file_size)},
                    )
                elapsed = time.time() - upload_start
                speed_mbps = file_size_mb / elapsed if elapsed > 0 else 0
                print(f"✅ Upload complete: {file_size_mb:.1f} MB in {elapsed:.0f}s ({speed_mbps:.1f} MB/s)")
                response.raise_for_status()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(f"⚠️  PUT failed (attempt {attempt + 1}): {exc}")

        if last_exc is not None:
            raise last_exc

        submit_request = ApiCreateSubmissionRequest()
        submit_request.competition_name = competition_name
        submit_request.blob_file_tokens = upload_response.token
        submit_request.submission_description = description

        response = self.client.competitions.competition_api_client.create_submission(submit_request)
        return response.ref

    def get_submission_score(self, submission_ref: str) -> Dict[str, Any]:
        from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest

        request = ApiGetSubmissionRequest()
        request.ref = submission_ref
        submission = self.client.competitions.competition_api_client.get_submission(request)
        return {
            "public_score": submission.public_score,
            "private_score": submission.private_score,
            "status": str(submission.status),
        }

    def wait_for_submission(self, submission_ref: str, timeout_minutes: int = 30) -> Dict[str, Any]:
        start = time.time()
        timeout = timeout_minutes * 60
        # Random initial offset so parallel workers don't all poll simultaneously
        time.sleep(random.uniform(5, 20))
        while time.time() - start < timeout:
            try:
                info = self.get_submission_score(submission_ref)
            except Exception as exc:
                if _is_retryable(exc):
                    print(f"⚠️  Polling error (will retry): {exc}")
                    time.sleep(30 + random.uniform(0, 20))
                    continue
                raise
            status = info.get("status", "")
            if "COMPLETE" in status:
                return info
            if "ERROR" in status:
                return info
            # Jittered poll interval: base 30s ± 10s so workers stay out of sync
            time.sleep(30 + random.uniform(-10, 10))
        final_info = self.get_submission_score(submission_ref)
        final_info["status"] = str(final_info.get("status", "TIMEOUT"))
        return final_info

@retry(wait=wait_random(10, 15))
def get_next_page(client, req):
    return client.competitions.competition_api_client.get_leaderboard(req)

@retry(wait=wait_random(2, 5), stop=stop_after_attempt(3))
def get_full_leaderboard(competition_name: str) -> Dict[str, Any]:
    """Fetch public and private leaderboard scores across pages."""
    import statistics
    from kagglesdk import KaggleClient
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

    def _fetch(public_flag: bool):
        client = KaggleClient()
        req = ApiGetLeaderboardRequest()
        req.competition_name = competition_name
        req.page_size = 100
        if public_flag:
            req.override_public = True
        entries = []
        seen = set()
        while True:
            # resp = client.competitions.competition_api_client.get_leaderboard(req)
            resp = get_next_page(client, req)
            entries.extend(resp.submissions)
            token = getattr(resp, "nextPageToken", None)
            token = token or getattr(resp, "next_page_token", None)
            if not token or token in seen:
                break
            req.page_token = token
            seen.add(token)
        return entries

    pub_entries = _fetch(public_flag=True)
    priv_entries = _fetch(public_flag=False)

    public_scores = []
    for sub in pub_entries:
        val = _safe_float(getattr(sub, "score", None))
        if val is not None:
            public_scores.append(val)

    private_scores = []
    for sub in priv_entries:
        val = _safe_float(getattr(sub, "score", None))
        if val is not None:
            private_scores.append(val)

    if private_scores == public_scores:
        private_scores = []

    return {
        "public_scores": public_scores,
        "private_scores": private_scores,
        "public_mean": statistics.mean(public_scores) if public_scores else 0,
        "public_median": statistics.median(public_scores) if public_scores else 0,
        "private_mean": statistics.mean(private_scores) if private_scores else 0,
        "private_median": statistics.median(private_scores) if private_scores else 0,
        "total_submissions": len(pub_entries),
    }



def get_leaderboard_rank_medal(leaderboard_scores, my_score):
    import math
    import statistics

    try:
        our_score = float(my_score)
    except (ValueError, TypeError):
        return -1, "none", 0.0, False

    if not leaderboard_scores:
        return 1, "gold", 100.0, True

    higher_is_better = leaderboard_scores[0] > leaderboard_scores[-1]
    combined = leaderboard_scores + [our_score]
    combined.sort(reverse=higher_is_better)
    rank = combined.index(our_score) + 1

    total = len(combined)
    def pct_ceil(p):
        return math.ceil(p * total)

    bronze_thr = total + 1
    silver_thr = total + 1
    gold_thr = total + 1

    if total <= 99:
        bronze_thr = pct_ceil(0.40)
        silver_thr = pct_ceil(0.20)
        gold_thr = pct_ceil(0.10)
    elif 100 <= total <= 249:
        bronze_thr = pct_ceil(0.40)
        silver_thr = pct_ceil(0.20)
        gold_thr = min(10, total)
    elif 250 <= total <= 999:
        bronze_thr = min(100, total)
        silver_thr = min(50, total)
        gold_thr = min(total, 10 + math.ceil(0.002 * total))
    else:
        bronze_thr = pct_ceil(0.10)
        silver_thr = pct_ceil(0.05)
        gold_thr = min(total, 10 + math.ceil(0.002 * total))

    if rank <= gold_thr:
        medal = "gold"
    elif rank <= silver_thr:
        medal = "silver"
    elif rank <= bronze_thr:
        medal = "bronze"
    else:
        medal = "none"

    total_others = len(leaderboard_scores)
    if total_others == 0:
        percentile = 100.0
        above_median = True
    else:
        if higher_is_better:
            num_worse = sum(1 for s in leaderboard_scores if s < our_score)
            num_equal = sum(1 for s in leaderboard_scores if s == our_score)
        else:
            num_worse = sum(1 for s in leaderboard_scores if s > our_score)
            num_equal = sum(1 for s in leaderboard_scores if s == our_score)
        percentile = 100.0 * (num_worse + 0.5 * num_equal) / total_others
        median_val = statistics.median(leaderboard_scores)
        above_median = our_score > median_val if higher_is_better else our_score < median_val

    return rank, medal, round(percentile, 6), above_median


def _load_offline_leaderboard_stats() -> Dict[str, Any]:
    """Load and combine offline leaderboard stats from JSON files."""
    import json
    from glob import glob
    from dsgym.datasets.config import REPO_ROOT
    
    offline_dir = REPO_ROOT / "data" / "DSPredict_Offline_Leaderboard"
    combined = {}
    for json_file in glob(str(offline_dir / "*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)
            combined.update(data)
    return combined


class KaggleSubmissionMetric(BaseMetric):
    """Metric that submits a Kaggle submission.csv and reports leaderboard-aware scores."""

    def __init__(self, timeout_minutes: int = 10, online: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.timeout_minutes = timeout_minutes
        self.online = online
        self._offline_stats = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "kaggle_submission"

    @property
    def requires_ground_truth(self) -> bool:
        return False

    # @retry(wait=wait_random(2, 5), stop=stop_after_attempt(5))
    def _submit_via_api(self, challenge_name, submission_path):
        """Submit file and wait for score, retrying the full flow with a fresh
        KaggleClient on every attempt.

        Stale gRPC/HTTP connections in KaggleClient produce
        ('Connection aborted.', BrokenPipeError) on the Kaggle API calls
        (start_submission_upload, create_submission, get_submission).
        Creating a new DSPredictSubmissionManager() on each retry gives a
        completely fresh connection pool, which clears the stale-connection issue.
        """
        last_exc = None
        for attempt in range(3):
            if attempt > 0:
                wait_secs = 20 * attempt + random.uniform(0, 10)
                print(f"⚠️  Submission attempt {attempt + 1} for {challenge_name}, retrying in {wait_secs:.0f}s…")
                time.sleep(wait_secs)
            try:
                # Fresh manager = fresh KaggleClient = fresh connection pool
                manager = DSPredictSubmissionManager()
                ref = manager.submit_file(
                    competition_name=challenge_name,
                    file_path=submission_path,
                    description=f"DSGym auto submission {challenge_name}",
                )
                score_info = manager.wait_for_submission(ref, timeout_minutes=self.timeout_minutes)
                return score_info
            except Exception as exc:
                if _is_retryable(exc):
                    last_exc = exc
                    print(f"⚠️  Submission attempt {attempt + 1} failed ({exc})")
                else:
                    raise
        raise last_exc
    def evaluate(
        self,
        prediction: str,
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Evaluate by submitting to Kaggle. The agent should return the path to the
        produced submission.csv in `prediction` (as KaggleReActAgent does).

        Expected kwargs in sample/extra_info via evaluator plumbing:
        - extra_info.challenge_name: Kaggle competition name.
        - extra_info.container_id and submission_dir: optional, used to locate files.
        """
        start = time.time()
        extra_info = kwargs.get("extra_info", {}) 
        print(extra_info)
        print(prediction)
        challenge_name = extra_info.get("challenge_name")
        
        # Get leaderboard stats (online or offline)
        if self.online:
            leaderboard_stats = get_full_leaderboard(challenge_name)
        else:
            if self._offline_stats is None:
                self._offline_stats = _load_offline_leaderboard_stats()
            leaderboard_stats = self._offline_stats.get(challenge_name, {})
        try:
            # Determine submission file path
            submission_path = (prediction or "").strip()
            if not submission_path or not os.path.exists(submission_path):
                return MetricResult(
                    metric_name=self.name,
                    score=None,
                    details={"reason": "submission file not found", "prediction": prediction, "leaderboard_stats": leaderboard_stats},
                )

            if not challenge_name:
                return MetricResult(
                    metric_name=self.name,
                    score=None,
                    details={"reason": "challenge_name not provided", "submission": submission_path, "leaderboard_stats": leaderboard_stats},
                )

            # Submit and wait
            score_info = self._submit_via_api(challenge_name, submission_path)
            status = str(score_info.pop("status", "")).upper()

            # Pull leaderboard stats and enrich
            pub = _safe_float(score_info.get("public_score"))
            priv = _safe_float(score_info.get("private_score"))
            if pub is not None and leaderboard_stats.get("public_scores"):
                r, m, p, above = get_leaderboard_rank_medal(leaderboard_stats["public_scores"], pub)
                score_info.update({
                    "public_rank": r,
                    "public_medal": m,
                    "public_percentile": p,
                    "public_above_median": above,
                })
            if priv is not None and leaderboard_stats.get("private_scores"):
                r, m, p, above = get_leaderboard_rank_medal(leaderboard_stats["private_scores"], priv)
                score_info.update({
                    "private_rank": r,
                    "private_medal": m,
                    "private_percentile": p,
                    "private_above_median": above,
                })

            details = {
                "status": status,
                "challenge_name": challenge_name,
                "local_submission_file": submission_path,
                "leaderboard_stats": leaderboard_stats,
                **score_info,
            }

            # Main metric score is the public score when available
            score = pub if pub is not None else None

            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details,
                evaluation_time=time.time() - start,
            )

        except Exception as e:
            print(f"Error computing KaggleSubmission Metrics: {e}, {challenge_name}")
            import traceback
            traceback.print_exc()
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"error": str(e), "leaderboard_stats": leaderboard_stats},
                error=str(e),
                evaluation_time=time.time() - start,
            )