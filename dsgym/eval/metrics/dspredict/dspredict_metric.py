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


def _is_kaggle_create_submission_forbidden(exc: BaseException) -> bool:
    """Return True when Kaggle rejects CreateSubmission with 403 Forbidden."""
    msg = str(exc).lower()
    return "403" in msg and "forbidden" in msg and "createsubmission" in msg


def _kaggle_403_hint(challenge_name: str) -> str:
    """Human-readable guidance for Kaggle CreateSubmission 403 errors."""
    hints = [
        "Kaggle account is authenticated but not permitted to submit to this competition.",
        "Common causes: competition closed, rules not accepted, not joined, team restrictions, or submission quota exceeded.",
        "Open the competition page in browser and verify: Join Competition + Rules accepted + Submit button enabled.",
    ]
    if challenge_name.startswith("playground-series-s"):
        hints.append(
            "This looks like a historical Playground competition; many older seasons are closed and no longer accept submissions."
        )
    return " ".join(hints)


def _extract_http_error_detail(exc: BaseException) -> str:
    """Best-effort extraction of HTTP response details from nested exceptions."""
    seen = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))

        response = getattr(cur, "response", None)
        if response is not None:
            text = getattr(response, "text", "")
            if text:
                return text[:2000]

        cause = getattr(cur, "__cause__", None)
        context = getattr(cur, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)
    return ""


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


class _ProgressWrapper:
    """File-like wrapper that prints upload progress.

    Works with ``requests.put(data=wrapper)`` — requests calls ``read(n)``
    repeatedly, which lets us track progress without loading the whole file
    into memory.

    Using a file-like object (not a generator) is critical: ``requests``/urllib3
    will honour a manually-set ``Content-Length`` header and will NOT add
    ``Transfer-Encoding: chunked``.  When httpx receives a generator it always
    adds chunked encoding even when ``Content-Length`` is also present, which
    sends conflicting headers to GCS and causes the server to drop the
    connection near the end of large uploads.
    """

    def __init__(self, file_path: str, file_size: int, chunk_size: int = 1024 * 1024):
        self._path = file_path
        self._file_size = file_size
        self._chunk_size = chunk_size
        self._uploaded = 0
        self._last_report = 0
        self._report_every = max(file_size // 20, chunk_size)
        self._fh = open(file_path, "rb")

    # requests calls read(n) with a fixed block size
    def read(self, n: int = -1) -> bytes:
        chunk = self._fh.read(self._chunk_size if n == -1 else n)
        if chunk:
            self._uploaded += len(chunk)
            if (self._uploaded - self._last_report >= self._report_every
                    or self._uploaded == self._file_size):
                pct = 100.0 * self._uploaded / self._file_size if self._file_size else 100
                print(
                    f"  … {self._uploaded / (1024*1024):.1f} /"
                    f" {self._file_size / (1024*1024):.1f} MB  ({pct:.0f}%)",
                    flush=True,
                )
                self._last_report = self._uploaded
        return chunk

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _make_upload_session():
    """Return a requests.Session with TCP keep-alive tuned for large uploads.

    TCP keepalive prevents NAT/firewall tables from expiring the connection
    during a slow, multi-minute upload — without it the remote end receives
    no data for the keepalive idle period and resets the TCP connection,
    producing ``BrokenPipeError``.
    """
    import socket
    import requests
    from requests.adapters import HTTPAdapter

    class _KeepAliveAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
            if hasattr(socket, "TCP_KEEPIDLE"):      # Linux
                opts += [
                    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30),
                    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
                    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6),
                ]
            elif hasattr(socket, "TCP_KEEPALIVE"):   # macOS
                opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))
            kwargs["socket_options"] = opts
            super().init_poolmanager(*args, **kwargs)

    session = requests.Session()
    session.mount("https://", _KeepAliveAdapter())
    return session


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

        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_size_mb = file_size / (1024 * 1024)

        upload_request = ApiStartSubmissionUploadRequest()
        upload_request.competition_name = competition_name
        upload_request.content_length = file_size
        upload_request.file_name = file_name

        upload_response = self.client.competitions.competition_api_client.start_submission_upload(
            upload_request
        )

        print(f"⬆️  Uploading {file_name} ({file_size_mb:.1f} MB) to {competition_name}…")
        upload_start = time.time()

        # Use requests (not httpx) with a file-like wrapper so that urllib3
        # honours the Content-Length header and does NOT add
        # Transfer-Encoding: chunked.  httpx always adds chunked encoding for
        # generator content even when Content-Length is provided, which sends
        # conflicting headers to GCS and causes the server to reset the
        # connection near the end of large uploads (BrokenPipeError).
        #
        # timeout=(30, None): 30 s to connect, then blocking for the upload
        # and response — no write-side timeout for slow connections.
        with _make_upload_session() as session, \
                _ProgressWrapper(file_path, file_size) as wrapper:
            response = session.put(
                upload_response.create_url,
                data=wrapper,
                headers={
                    "Content-Length": str(file_size),
                    "Content-Type": "application/octet-stream",
                },
                timeout=(30, None),
            )

        elapsed = time.time() - upload_start
        speed_mbps = file_size_mb / elapsed if elapsed > 0 else 0
        print(f"✅ Upload complete: {file_size_mb:.1f} MB in {elapsed:.0f}s ({speed_mbps:.1f} MB/s)")
        response.raise_for_status()

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
            "error_description": submission.error_description or "",
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
                err = info.get("error_description", "")
                if err:
                    print(f"⚠️  Kaggle error: {err}")
                return info
            # Jittered poll interval: base 30s ± 10s so workers stay out of sync
            time.sleep(30 + random.uniform(-10, 10))
        final_info = self.get_submission_score(submission_ref)
        final_info["status"] = str(final_info.get("status", "TIMEOUT"))
        return final_info


def _validate_submission(submission_path: str, challenge_name: str) -> list:
    """Validate a submission CSV against the local sample submission.

    Returns a list of human-readable error strings (empty = valid).
    Looks for ``sample_submission*.csv`` in any known data split under
    ``data/data/dspredict-{split}/{challenge_name}/``.

    The errors are injected back into the agent's conversation so it can
    self-correct before the file is uploaded to Kaggle.
    """
    import csv
    import glob as _glob

    try:
        from dsgym.datasets.config import RAW_DATA_DIR
        base = RAW_DATA_DIR
    except Exception:
        return []

    sample_path = None
    for split in ("sampled", "easy", "hard", "lite"):
        pattern = str(base / f"dspredict-{split}" / challenge_name / "sample_submission*.csv")
        matches = [p for p in _glob.glob(pattern) if not p.endswith(".zip")]
        if matches:
            sample_path = matches[0]
            break

    if sample_path is None:
        return []

    errors = []

    try:
        with open(sample_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            expected_cols = next(reader, None)
            expected_rows = sum(1 for _ in reader)
    except Exception as e:
        return [f"Could not read sample submission: {e}"]

    if not expected_cols:
        return []

    try:
        with open(submission_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            actual_cols = next(reader, None)
            actual_rows = sum(1 for _ in reader)
    except Exception as e:
        return [f"Could not read submission file: {e}"]

    if not actual_cols:
        return ["Submission file appears to be empty or has no header"]

    missing = sorted(set(expected_cols) - set(actual_cols))
    extra   = sorted(set(actual_cols)   - set(expected_cols))

    if missing:
        errors.append(f"Missing required columns: {missing}")
    if extra:
        errors.append(f"Extra/unexpected columns (remove these): {extra}")
    if actual_rows != expected_rows:
        errors.append(
            f"Wrong number of rows: expected {expected_rows:,}, got {actual_rows:,}"
        )

    return errors


def _find_sample_submission_path(challenge_name: str) -> Optional[str]:
    """Locate sample_submission.csv for the given challenge across known splits."""
    import glob as _glob

    try:
        from dsgym.datasets.config import RAW_DATA_DIR
        base = RAW_DATA_DIR
    except Exception:
        return None

    for split in ("sampled", "easy", "hard", "lite"):
        pattern = str(base / f"dspredict-{split}" / challenge_name / "sample_submission*.csv")
        matches = [p for p in _glob.glob(pattern) if not p.endswith(".zip")]
        if matches:
            return matches[0]
    return None


def _auto_repair_submission(submission_path: str, challenge_name: str) -> Dict[str, Any]:
    """Attempt to coerce a malformed submission into a valid Kaggle format.

    Repair strategy:
    1) Preserve model outputs when possible (fix column names/order and row count).
    2) If irreparable, fall back to copying sample_submission.csv to guarantee validity.
    """
    info: Dict[str, Any] = {"repaired": False, "notes": []}
    sample_path = _find_sample_submission_path(challenge_name)
    info["sample_submission_path"] = sample_path
    if not sample_path:
        info["reason"] = "sample_submission_not_found"
        return info

    backup_path = f"{submission_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(submission_path, backup_path)
        info["backup_path"] = backup_path
    except Exception as exc:
        info["notes"].append(f"Could not create backup: {exc}")

    try:
        import pandas as pd

        sample_df = pd.read_csv(sample_path)
        expected_cols = list(sample_df.columns)

        try:
            sub_df = pd.read_csv(submission_path)
        except Exception as exc:
            info["notes"].append(f"Submission unreadable ({exc}); using sample fallback")
            shutil.copy2(sample_path, submission_path)
            post_errors = _validate_submission(submission_path, challenge_name)
            info["repaired"] = len(post_errors) == 0
            info["strategy"] = "fallback_copy_sample"
            info["post_errors"] = post_errors
            return info

        # Fix columns first.
        if list(sub_df.columns) != expected_cols:
            if len(sub_df.columns) == len(expected_cols):
                sub_df.columns = expected_cols
                info["notes"].append("Renamed submission columns to expected header")
            else:
                repaired = pd.DataFrame()
                # Keep canonical ID column from sample file.
                repaired[expected_cols[0]] = sample_df[expected_cols[0]].iloc[: len(sub_df)].reset_index(drop=True)
                candidate_cols = [
                    c for c in sub_df.columns
                    if c.lower() not in {expected_cols[0].lower(), "id"}
                ]
                if not candidate_cols and len(sub_df.columns) > 0:
                    candidate_cols = [sub_df.columns[-1]]

                for i, col in enumerate(expected_cols[1:]):
                    if candidate_cols:
                        src = candidate_cols[min(i, len(candidate_cols) - 1)]
                        repaired[col] = sub_df[src]
                    else:
                        repaired[col] = sample_df[col].iloc[: len(sub_df)].reset_index(drop=True)

                sub_df = repaired
                info["notes"].append("Rebuilt submission columns to match sample schema")

        # Fix row count.
        expected_rows = len(sample_df)
        if len(sub_df) > expected_rows:
            sub_df = sub_df.iloc[:expected_rows].copy()
            info["notes"].append("Trimmed extra rows to expected sample length")
        elif len(sub_df) < expected_rows:
            pad = sample_df.iloc[len(sub_df):].copy().reset_index(drop=True)
            # Preserve last prediction value for padded rows when available.
            if len(sub_df) > 0:
                for col in expected_cols[1:]:
                    if col in sub_df.columns:
                        pad[col] = sub_df[col].iloc[-1]
            sub_df = pd.concat([sub_df, pad], ignore_index=True)
            info["notes"].append("Padded missing rows to expected sample length")

        sub_df = sub_df.reindex(columns=expected_cols)

        # Fill any residual NaNs.
        if sub_df.isna().any().any():
            for col in expected_cols:
                if sub_df[col].isna().any():
                    fallback = sample_df[col] if col in sample_df.columns else 0
                    sub_df[col] = sub_df[col].fillna(fallback)
            info["notes"].append("Filled NaN values in repaired submission")

        sub_df.to_csv(submission_path, index=False)

        post_errors = _validate_submission(submission_path, challenge_name)
        if post_errors:
            info["notes"].append("Repair still invalid; falling back to sample submission")
            shutil.copy2(sample_path, submission_path)
            post_errors = _validate_submission(submission_path, challenge_name)
            info["strategy"] = "fallback_copy_sample"
            info["post_errors"] = post_errors
            info["repaired"] = len(post_errors) == 0
            return info

        info["strategy"] = "schema_and_rowcount_repair"
        info["post_errors"] = []
        info["repaired"] = True
        return info

    except Exception as exc:
        info["reason"] = f"repair_exception: {exc}"
        try:
            shutil.copy2(sample_path, submission_path)
            post_errors = _validate_submission(submission_path, challenge_name)
            info["strategy"] = "repair_exception_fallback_copy_sample"
            info["post_errors"] = post_errors
            info["repaired"] = len(post_errors) == 0
        except Exception as fallback_exc:
            info["fallback_error"] = str(fallback_exc)
        return info

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
        for attempt in range(5):
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
                elif _is_kaggle_create_submission_forbidden(exc):
                    detail = _extract_http_error_detail(exc)
                    raise PermissionError(
                        f"Kaggle CreateSubmission forbidden (403) for {challenge_name}. "
                        f"{_kaggle_403_hint(challenge_name)}"
                        + (f" Response detail: {detail}" if detail else "")
                    ) from exc
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

            # Validate and auto-repair malformed submissions before upload.
            validation_errors_pre = _validate_submission(submission_path, challenge_name)
            repair_info: Dict[str, Any] = {}
            if validation_errors_pre:
                print(f"⚠️  Submission format errors detected pre-upload: {validation_errors_pre}")
                repair_info = _auto_repair_submission(submission_path, challenge_name)
                validation_errors_post = _validate_submission(submission_path, challenge_name)
                if validation_errors_post:
                    return MetricResult(
                        metric_name=self.name,
                        score=None,
                        details={
                            "reason": "submission invalid after auto-repair",
                            "challenge_name": challenge_name,
                            "local_submission_file": submission_path,
                            "validation_errors_pre": validation_errors_pre,
                            "validation_errors_post": validation_errors_post,
                            "repair_info": repair_info,
                            "leaderboard_stats": leaderboard_stats,
                        },
                    )
            else:
                validation_errors_post = []

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
                "validation_errors_pre": validation_errors_pre,
                "validation_errors_post": validation_errors_post,
                "repair_info": repair_info,
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
            details = {"error": str(e), "leaderboard_stats": leaderboard_stats}
            if _is_kaggle_create_submission_forbidden(e):
                details["reason"] = "kaggle_create_submission_forbidden"
                details["hint"] = _kaggle_403_hint(challenge_name or "")
                http_detail = _extract_http_error_detail(e)
                if http_detail:
                    details["http_error_detail"] = http_detail
            return MetricResult(
                metric_name=self.name,
                score=None,
                details=details,
                error=str(e),
                evaluation_time=time.time() - start,
            )