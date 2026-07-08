"""
DSGym Results Dashboard
Streamlit app combining analysis.ipynb + dag_consistency_analysis.ipynb
Run from repo root or evaluation_results/:
    streamlit run evaluation_results/app.py
"""

import hashlib
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from rouge_score import rouge_scorer as rouge_scorer_lib

matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent

N_PROBLEMS = 54

COLORS = {
    "dags": "#2196F3",
    "nodags": "#FF9800",
    "dags_adaptive-revision": "#4CAF50",
}
# Human-readable labels for known model IDs (extended on the fly by discovery)
MODEL_LABELS = {
    "claude-sonnet-4-5":  "Claude Sonnet 4.5",
    "claude-sonnet-4-6":  "Claude Sonnet 4.6",
    "sonnet-4-5":         "Sonnet 4.5",
    "gpt-5.1":            "GPT-5.1",
    "gpt-4o-mini":        "GPT-4o Mini",
    "gpt-4o":             "GPT-4o",
    "deepseek-v4-pro":    "DeepSeek V4 Pro",
    "kimi-k2p7-code":     "Kimi K2 Code",
    "Qwen3-Coder-480B":   "Qwen3 Coder 480B",
    "Qwen3-Coder-480B-A35B-Instruct":     "Qwen3-Coder 480B",
    "Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen3-Coder 480B FP8",
}
MEDAL_COLORS = {"gold": "#FFD700", "silver": "#C0C0C0", "bronze": "#CD7F32", "none": "#DDDDDD"}
MEDAL_TYPES = ["gold", "silver", "bronze"]
MEDAL_TYPES_FULL = ["gold", "silver", "bronze", "none"]
MEDAL_LABELS_FULL = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze", "none": "No Medal"}
STOP_WORDS = {
    "the","a","an","and","or","to","of","in","for","with","on","from","using",
    "use","all","each","by","into","as","at","is","are","be","its","this","that",
    "if","any","no","data","model","step","based","new","make","get","set","run",
    "then","also","both","only","when","will","has","have","not","their",
}

# ── Color palette: consistent hash-based assignment per model ─────────────────
_COLOR_PALETTE = [
    "#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336",
    "#00BCD4","#FF5722","#607D8B","#795548","#E91E63",
    "#3F51B5","#009688","#FFC107","#8BC34A","#673AB7",
    "#03A9F4","#CDDC39","#FF7043","#26A69A","#7E57C2",
]

def _model_color(model: str) -> str:
    """Deterministic colour for any model name."""
    idx = int(hashlib.md5(model.encode()).hexdigest(), 16) % len(_COLOR_PALETTE)
    return _COLOR_PALETTE[idx]

# ── Auto-discovery: results files ─────────────────────────────────────────────
_KNOWN_MODES     = ["dags_adaptive-revision", "dags", "nodags"]
_KNOWN_PROVIDERS = ["fireworks_ai_", "fireworks_Qwen_", "fireworks_", "openai_"]

def _discover_results_files():
    """
    Scan RESULTS_DIR for *_results.json files and build
    (dp_runs, dp_hard_runs, bio_runs) dicts dynamically.
    Files without an explicit mode suffix are treated as 'nodags'
    and only included when no explicit nodags file exists for the same key.
    """
    # Collect explicit-mode files first, implicit-mode files second
    explicit: dict[str, dict] = {"dp": {}, "dp_hard": {}, "bio": {}}
    implicit: dict[str, dict] = {"dp": {}, "dp_hard": {}, "bio": {}}

    for fpath in sorted(RESULTS_DIR.glob("*_results.json")):
        name = fpath.name
        if name.startswith("daeval_"):
            continue
        m = re.match(r'^(.+?)_litellm_(.+?)_results\.json$', name)
        if not m:
            continue
        benchmark, rest = m.group(1), m.group(2)

        # Determine benchmark bucket
        if benchmark == "dspredict-sampled":
            bucket = "dp"
        elif benchmark == "dspredict-hard":
            bucket = "dp_hard"
        elif benchmark in ("dsbio-sampled", "dsbio"):
            bucket = "bio"
        else:
            continue  # unknown benchmark

        # Extract mode from end of rest
        mode = None
        for candidate in _KNOWN_MODES:
            if rest.endswith(f"_{candidate}"):
                mode = candidate
                rest = rest[: -len(f"_{candidate}")]
                break

        # Strip provider prefix
        model = rest
        for provider in _KNOWN_PROVIDERS:
            if model.startswith(provider):
                model = model[len(provider):]
                break

        # Embed variant into model name so adaptive runs appear as a distinct model
        if mode == "dags_adaptive-revision":
            model = f"{model} (adaptive)"

        key = (model, mode if mode else "nodags")
        if mode is not None:
            explicit[bucket][key] = fpath
        else:
            implicit[bucket][key] = fpath

    # Merge: explicit wins; implicit fills gaps
    result = {}
    for bucket in ("dp", "dp_hard", "bio"):
        merged = dict(implicit[bucket])
        merged.update(explicit[bucket])
        result[bucket] = merged

    return result["dp"], result["dp_hard"], result["bio"]


# ── Auto-discovery: per-task DAG files ───────────────────────────────────────
def _discover_dag_model_configs():
    """
    Scan RESULTS_DIR for per-task *_dags*.json files and build
    DAG_MODEL_CONFIGS: {model_name: {suffix, color}}.
    """
    slug_set: set[str] = set()
    for fpath in RESULTS_DIR.iterdir():
        if fpath.suffix != ".json":
            continue
        # Per-task files: {comp}_{num}_{slug}.json  (num is a digit sequence)
        m = re.match(r'^.+?_\d+_(.+)$', fpath.stem)
        if not m:
            continue
        slug = m.group(1)
        # Only collect DAG-related slugs
        if slug.endswith("_dags_adaptive-revision") or slug.endswith("_dags"):
            slug_set.add(slug)

    def _slug_to_name(raw: str) -> str:
        name = raw.replace("_", "-")
        # Only fix GPT decimal versions: gpt-5-1 → gpt-5.1, gpt-4-0 → gpt-4.0
        name = re.sub(r'\bgpt-(\d+)-(\d+)\b', r'gpt-\1.\2', name)
        return name

    configs: dict[str, dict] = {}
    for slug in sorted(slug_set):
        if slug.endswith("_dags_adaptive-revision"):
            base = _slug_to_name(slug[: -len("_dags_adaptive-revision")])
            model_name = f"{base} (adaptive)"
        else:
            base = _slug_to_name(slug[: -len("_dags")])
            model_name = base
        configs[model_name] = {
            "suffix": f"_{slug}.json",
            "color":  _model_color(base),   # colour keyed on base name so adaptive shares colour
        }
    return configs


def _patch_per_task_item(raw: dict, fname: str) -> dict:
    """Fill in challenge_name from filename when it's missing in a per-task file."""
    m = re.match(r'^(.+?)_\d+_.+$', fname)
    comp = m.group(1) if m else None
    ks      = (raw.get("metrics") or {}).get("kaggle_submission") or {}
    details = ks.get("details") or {}
    if not details.get("challenge_name") and comp:
        raw     = dict(raw)
        raw["metrics"] = dict(raw.get("metrics") or {})
        raw["metrics"]["kaggle_submission"] = dict(ks)
        raw["metrics"]["kaggle_submission"]["details"] = dict(details)
        raw["metrics"]["kaggle_submission"]["details"]["challenge_name"] = comp
    return raw


def _discover_per_task_runs(existing_dp, existing_dp_hard, existing_bio):
    """
    For each per-task DAG slug whose data is NOT already covered by a results file,
    synthesise a list-of-Paths entry in the appropriate run dict.
    Returns supplementary (dp_extra, dp_hard_extra, bio_extra) dicts.
    """
    slug_files: dict[str, list[Path]] = {}
    for fpath in sorted(RESULTS_DIR.iterdir()):
        if fpath.suffix != ".json": continue
        m = re.match(r'^.+?_\d+_(.+)$', fpath.stem)
        if not m: continue
        slug = m.group(1)
        if slug.endswith("_dags_adaptive-revision") or slug.endswith("_dags"):
            slug_files.setdefault(slug, []).append(fpath)

    dp_extra, dp_hard_extra, bio_extra = {}, {}, {}

    for slug, files in slug_files.items():
        file_suffix = f"_{slug}.json"
        model_name = next(
            (m for m, cfg in DAG_MODEL_CONFIGS.items() if cfg["suffix"] == file_suffix),
            None,
        )
        if model_name is None:
            continue
        mode = "dags_adaptive-revision" if slug.endswith("_dags_adaptive-revision") else "dags"
        key  = (model_name, mode)

        # Skip if already covered by a results file in ALL three buckets
        # (must check per-bucket so a sampled file doesn't suppress a hard per-task entry)
        covered_dp   = key in existing_dp
        covered_hard = key in existing_dp_hard
        covered_bio  = key in existing_bio

        # Peek at first file to determine benchmark from sample_id
        try:
            sample = json.load(open(files[0]))
        except Exception:
            continue
        sid = sample.get("sample_id", "") or ""
        if sid.startswith("dsbio_"):
            if not covered_bio:
                bio_extra[key] = files
        elif not covered_hard:
            dp_hard_extra[key] = files   # default to hard; sampled already has results files

    return dp_extra, dp_hard_extra, bio_extra


DP_RUNS, DP_HARD_RUNS, BIO_RUNS = _discover_results_files()
DAG_MODEL_CONFIGS = _discover_dag_model_configs()
_dp_ex, _dp_hard_ex, _bio_ex = _discover_per_task_runs(DP_RUNS, DP_HARD_RUNS, BIO_RUNS)
DP_RUNS.update(_dp_ex)
DP_HARD_RUNS.update(_dp_hard_ex)
BIO_RUNS.update(_bio_ex)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
SKIP_LB = {"public_scores","private_scores","scores"}

def _parse_dp_item(item):
    ks      = item.get("metrics",{}).get("kaggle_submission",{})
    details = ks.get("details") or {}
    lb      = details.get("leaderboard_stats",{}) if isinstance(details, dict) else {}
    for k in SKIP_LB:
        lb.pop(k, None)
    tok = item.get("token_usage") or {}
    return {
        "sample_id":      item.get("sample_id"),
        "challenge":      details.get("challenge_name") if isinstance(details,dict) else None,
        "status":         (details.get("status","") if isinstance(details,dict) else ""),
        "score":          ks.get("score"),
        "public_medal":   details.get("public_medal","none") if isinstance(details,dict) else "none",
        "private_medal":  details.get("private_medal","none") if isinstance(details,dict) else "none",
        "pub_percentile": details.get("public_percentile") if isinstance(details,dict) else None,
        "priv_percentile":details.get("private_percentile") if isinstance(details,dict) else None,
        "pub_above_med":  details.get("public_above_median")  if isinstance(details,dict) else None,
        "pub_rank":       details.get("public_rank") if isinstance(details,dict) else None,
        "execution_time": item.get("execution_time"),
        "total_turns":    item.get("total_turns"),
        "success":        item.get("success"),
        "trajectory":     item.get("trajectory",[]),
        "input_tokens":   tok.get("input_tokens"),
        "output_tokens":  tok.get("output_tokens"),
        "total_tokens":   tok.get("total_tokens"),
    }


def _load_run_dp(val) -> list:
    """Load DSPredict items from a results JSON or a list of per-task JSON paths."""
    if isinstance(val, Path):
        if not val.exists(): return []
        with open(val) as f:
            return [_parse_dp_item(i) for i in json.load(f)]
    # Per-task files: deduplicate by comp (take first/lowest-index file per comp)
    slug_re = re.compile(r"^(.+?)_\d+_")
    seen_comp: set[str] = set()
    items = []
    for p in sorted(val):   # sorted → lowest index first
        m = slug_re.match(p.name)
        comp = m.group(1) if m else p.name
        if comp in seen_comp:
            continue
        seen_comp.add(comp)
        try:
            raw = json.load(open(p))
            items.append(_parse_dp_item(_patch_per_task_item(raw, p.name)))
        except Exception:
            pass
    return items


def _load_run_bio(val) -> list:
    """Load DSBio items from a results JSON or a list of per-task JSON paths."""
    def _parse_bio(item):
        em = item.get("metrics",{}).get("exact_match",{})
        return {
            "sample_id":      item.get("sample_id"),
            "exact_match":    em.get("score"),
            "execution_time": item.get("execution_time"),
            "total_turns":    item.get("total_turns"),
            "success":        item.get("success"),
            "query_id":       item.get("extra_info",{}).get("query_id"),
            "trajectory":     item.get("trajectory",[]),
        }

    if isinstance(val, Path):
        if not val.exists(): return []
        with open(val) as f:
            return [_parse_bio(i) for i in json.load(f)]
    items = []
    for p in val:
        try:
            items.append(_parse_bio(json.load(open(p))))
        except Exception:
            pass
    return items


@st.cache_data(show_spinner=False)
def load_all_data():
    dp_data, dp_hard_data, bio_data = {}, {}, {}
    for key, val in DP_RUNS.items():
        rows = _load_run_dp(val)
        if rows: dp_data[key] = rows
    for key, val in DP_HARD_RUNS.items():
        rows = _load_run_dp(val)
        if rows: dp_hard_data[key] = rows
    for key, val in BIO_RUNS.items():
        rows = _load_run_bio(val)
        if rows: bio_data[key] = rows
    return dp_data, dp_hard_data, bio_data


def _find_json_objects(text):
    results, depth, start, in_string, escape_next = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if escape_next:   escape_next = False; continue
        if ch == "\\" and in_string: escape_next = True; continue
        if ch == '"':     in_string = not in_string; continue
        if in_string:     continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                results.append(text[start:i+1]); start = -1
    return results


def _extract_dags_from_content(content, turn_index=0):
    """Extract DAG dicts from a single turn's content string.

    Handles:
      1. <reasoning_dag> / <implementation_dag> XML tags (DeepSeek adaptive)
      2. Closed fences with dag_type field (GPT, Claude)
      3. Unclosed fences (DeepSeek standard — fence never closed)
      4. Bare JSON anywhere in content
    """
    dags, seen = [], set()

    def _add(candidate, forced_dag_type=None):
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict): return
            if forced_dag_type:
                obj["dag_type"] = forced_dag_type
            if "dag_type" not in obj or "nodes" not in obj: return
            key = json.dumps(obj, sort_keys=True)
            if key not in seen:
                seen.add(key)
                obj["_turn_index"] = turn_index
                dags.append(obj)
        except Exception:
            pass

    # 1. XML tag format: <reasoning_dag>```json ... ```</reasoning_dag>
    for tag, dag_type in [("reasoning_dag", "reasoning"), ("implementation_dag", "implementation")]:
        for m in re.finditer(rf"<{tag}>([\s\S]*?)</{tag}>", content):
            inner = m.group(1).strip()
            # strip optional fences inside the tag
            fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", inner)
            if fence_m:
                inner = fence_m.group(1)
            for candidate in _find_json_objects(inner):
                _add(candidate, forced_dag_type=dag_type)

    # 2. Closed fences (with dag_type field)
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", content):
        for candidate in _find_json_objects(m.group(1)):
            _add(candidate)

    # 3. Unclosed fence (DeepSeek standard): ```json ... <EOF>
    m = re.search(r"```(?:json)?\s*([\s\S]+)$", content)
    if m and "```" not in m.group(1):
        for candidate in _find_json_objects(m.group(1)):
            _add(candidate)

    # 4. Bare JSON anywhere in content
    for candidate in _find_json_objects(content):
        _add(candidate)

    return dags


def _extract_from_traj(traj):
    all_dags, code_blocks = [], []
    for turn_idx, turn in enumerate(traj):
        if not isinstance(turn, dict) or turn.get("role") != "assistant": continue
        content = turn.get("content","")
        all_dags.extend(_extract_dags_from_content(content, turn_idx))
        code_blocks.extend(re.findall(r"<python>(.*?)</python>", content, re.DOTALL))
    return all_dags, code_blocks


def _strip_dag_and_code(content: str) -> str:
    """Return only the natural-language reasoning from an assistant turn.
    Strips DAG blocks (closed and unclosed), code fences, and <python> tags.
    """
    # Remove closed XML dag/code blocks
    for tag in ("reasoning_dag", "implementation_dag", "python", "implementation"):
        content = re.sub(rf"<{tag}>[\s\S]*?</{tag}>", "", content)
    # Remove unclosed <reasoning_dag> / <implementation_dag> (DeepSeek adaptive style)
    for tag in ("reasoning_dag", "implementation_dag"):
        content = re.sub(rf"<{tag}>[\s\S]*", "", content)
    # Remove code fences (```lang ... ```)
    content = re.sub(r"```[\w]*[\s\S]*?```", "", content)
    # If <reasoning> tag present, extract just that
    m = re.search(r"<reasoning>([\s\S]*?)</reasoning>", content)
    if m:
        return m.group(1).strip()
    # Strip remaining XML-like tags
    content = re.sub(r"<[^>]+>[\s\S]*?</[^>]+>", "", content)
    content = re.sub(r"<[^>]+>", "", content)
    return content.strip()


def _node_labels(dag):
    return [n.get("label","").lower() for n in dag.get("nodes",[])]


def _keywords(labels):
    words = set()
    for lbl in labels:
        for w in re.findall(r"[a-z][a-z0-9]+", lbl.lower()):
            if w not in STOP_WORDS and len(w) > 2: words.add(w)
    return words


def _jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)


def _rouge(labels_a, labels_b):
    scorer = rouge_scorer_lib.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    a, b = " ".join(labels_a), " ".join(labels_b)
    if not a or not b: return {"rouge1":0.0,"rouge2":0.0,"rougeL":0.0}
    s = scorer.score(a, b)
    return {"rouge1":s["rouge1"].fmeasure,"rouge2":s["rouge2"].fmeasure,"rougeL":s["rougeL"].fmeasure}


def _build_dag_record(item: dict, model: str, benchmark: str, comp: str, fname: str) -> dict | None:
    """Convert a raw JSON item into a DAG consistency record. Returns None if no trajectory."""
    traj = item.get("trajectory", [])
    if not traj:
        return None

    # Use the same extractor as the DAG Visualizer so parsing is consistent
    all_dags = _extract_embedded_dags(traj)
    r_dags = [d for d in all_dags if d.get("dag_type") == "reasoning"]
    i_dags = [d for d in all_dags if "implementation" in d.get("dag_type", "")]
    has_dag    = bool(r_dags)
    has_ri_dag = bool(r_dags) and bool(i_dags)

    # Code blocks: support both <python> tags (GPT/Claude) and ```python fences (DeepSeek)
    code_blocks = []
    for turn in traj:
        if turn.get("role") != "assistant": continue
        content = turn.get("content", "") or ""
        code_blocks.extend(re.findall(r"<python>(.*?)</python>", content, re.DOTALL))
        code_blocks.extend(re.findall(r"```python\s*(.*?)\s*```", content, re.DOTALL))

    status  = (item.get("metrics", {}).get("kaggle_submission", {})
                   .get("details", {}).get("status", "") or "")
    outcome = ("complete" if "COMPLETE" in status else
               "error"    if "ERROR"    in status else "no_submit")
    all_code = " ".join(code_blocks).lower()

    rec: dict = {
        "model": model, "comp": comp, "fname": fname, "benchmark": benchmark,
        "outcome": outcome, "has_dag": has_dag, "has_ri_dag": has_ri_dag,
        "all_code": all_code, "trajectory": traj,
        "all_r_dags": r_dags, "all_i_dags": i_dags,
        "n_reasoning_dags": len(r_dags), "n_implementation_dags": len(i_dags),
        "total_turns": item.get("total_turns"),
        "exec_min": (item.get("execution_time") or 0) / 60,
        "r_kw": set(), "i_kw": set(), "r_lbls": [], "i_lbls": [],
    }

    # Populate reasoning labels whenever we have reasoning dags (not just when has_ri_dag)
    if r_dags:
        r_lbls = _node_labels(r_dags[0])
        r_kw   = _keywords(r_lbls)
        rec.update({"r_lbls": r_lbls, "r_kw": r_kw, "r_nodes": len(r_lbls)})

    if has_ri_dag:
        i_lbls = _node_labels(i_dags[0])
        i_kw   = _keywords(i_lbls)
        r_lbls = rec["r_lbls"]
        r_kw   = rec["r_kw"]
        ri_jac = _jaccard(r_kw, i_kw)
        ri_ov  = len(r_kw & i_kw) / max(1, len(r_kw))
        code_set  = set(re.findall(r"[a-z][a-z0-9]+", all_code))
        node_hits = [bool(_keywords([lbl]) & code_set) for lbl in i_lbls]
        code_cov  = sum(node_hits) / max(1, len(node_hits))
        misses    = [i / len(node_hits) for i, h in enumerate(node_hits) if not h]
        rouge = _rouge(r_lbls, i_lbls)
        rec.update({
            "i_nodes": len(i_lbls), "node_delta": len(i_lbls) - len(r_lbls),
            "ri_jaccard": ri_jac, "ri_overlap": ri_ov,
            "ri_rouge1": rouge["rouge1"], "ri_rouge2": rouge["rouge2"], "ri_rougeL": rouge["rougeL"],
            "code_cov": code_cov, "first_miss": misses[0] if misses else None,
            "i_kw": i_kw, "i_lbls": i_lbls,
        })
    return rec


@st.cache_data(show_spinner=False)
def load_dag_consistency_records():
    """Load DAG consistency records from per-task files, one record per (model, comp)."""
    records: list[dict] = []
    slug_re = re.compile(r"^(.+?)_\d+_")

    for model, cfg in DAG_MODEL_CONFIGS.items():
        suffix = cfg["suffix"]
        seen_comp: set[str] = set()
        for fpath in sorted(RESULTS_DIR.iterdir()):
            if not fpath.name.endswith(suffix): continue
            m = slug_re.match(fpath.name)
            if not m: continue
            comp = m.group(1)
            if comp in seen_comp: continue   # take first file per comp (sorted → lowest index)
            seen_comp.add(comp)
            try:
                item = json.load(open(fpath))
            except Exception:
                continue
            if not isinstance(item, dict): continue
            sid = item.get("sample_id", "") or ""
            benchmark = "DSBio" if sid.startswith("dsbio_") else "DSPredict"
            rec = _build_dag_record(item, model, benchmark, comp, fpath.name)
            if rec:
                records.append(rec)

    return records


@st.cache_data(show_spinner=False)
def load_drift_records():
    """Load per-task traces for ALL model/mode combos (dags + nodags + adaptive).
    Returns list of dicts with model, base_model, mode, comp, success, trajectory.
    Takes first file (lowest index) per (model_full, comp) pair.
    """
    records: list[dict] = []
    seen: set[tuple] = set()
    slug_re = re.compile(r"^(.+?)_(\d+)_(.+)$")

    def _slug_to_name(raw: str) -> str:
        name = raw.replace("_", "-")
        name = re.sub(r"\bgpt-(\d+)-(\d+)\b", r"gpt-\1.\2", name)
        return name

    for fpath in sorted(RESULTS_DIR.iterdir()):
        if fpath.suffix != ".json": continue
        m = slug_re.match(fpath.stem)
        if not m: continue
        comp, slug = m.group(1), m.group(3)

        mode = None
        slug_base = slug
        for candidate in ["dags_adaptive-revision", "dags", "nodags"]:
            if slug.endswith(f"_{candidate}"):
                mode = candidate
                slug_base = slug[: -len(f"_{candidate}")]
                break
        if mode is None: continue

        base_model = _slug_to_name(slug_base)
        suffix = {"dags_adaptive-revision": " (adaptive)", "nodags": " (nodags)", "dags": " (dags)"}[mode]
        model_full = f"{base_model}{suffix}"

        key = (model_full, comp)
        if key in seen: continue
        seen.add(key)

        try:
            item = json.load(open(fpath))
            if not isinstance(item, dict): continue
            traj = item.get("trajectory", [])
            if not traj: continue
            status = ((item.get("metrics") or {}).get("kaggle_submission") or {}) \
                         .get("details", {}).get("status", "") or ""
            records.append({
                "model": model_full,
                "base_model": base_model,
                "mode": mode,
                "comp": comp,
                "success": "COMPLETE" in status,
                "trajectory": traj,
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Failure pattern analysis
# ─────────────────────────────────────────────────────────────────────────────

_FMT_ERROR_RE    = re.compile(r"No python code found", re.I)
_TIMEOUT_RE      = re.compile(r"timed out|time.?out", re.I)
_SERVER_ERROR_RE = re.compile(r"500 Internal Server|503 Service|Server disconnected", re.I)
_KERNEL_RE       = re.compile(r"KernelError", re.I)
_TRACEBACK_RE    = re.compile(r"Traceback \(most recent call last\)|Error:\s", re.I)
_PYTHON_OPEN_RE  = re.compile(r"<python>|```python", re.I)
_PYTHON_CLOSE_RE = re.compile(r"</python>|```\s*$", re.M)
_INFO_RE         = re.compile(r"<information>(.*?)</information>", re.S)


def _analyze_trajectory(traj: list) -> dict:
    """Extract structured failure-pattern stats from a trajectory."""
    fmt_errors = timeouts = server_errors = kernel_errors = 0
    runtime_errors = truncations = empty_responses = 0
    first_turn_code = first_turn_dag = False
    prev_was_fmt_error = False
    recoveries = fmt_error_chances = 0
    asst_seen = 0

    for turn in traj:
        if not isinstance(turn, dict):
            continue
        role    = turn.get("role", "")
        content = turn.get("content", "") or ""

        if role == "assistant":
            # First assistant turn flags
            if asst_seen == 0:
                first_turn_code = bool(_PYTHON_OPEN_RE.search(content))
                first_turn_dag  = bool(re.search(r'"dag_type"', content) or
                                       re.search(r"<reasoning_dag>", content))
            asst_seen += 1

            # Empty response
            if len(content.strip()) < 60:
                empty_responses += 1

            # Truncation: opens <python> but never closes it
            if _PYTHON_OPEN_RE.search(content) and not _PYTHON_CLOSE_RE.search(content):
                truncations += 1

            # Recovery tracking
            if prev_was_fmt_error:
                fmt_error_chances += 1
                if _PYTHON_OPEN_RE.search(content):
                    recoveries += 1
                prev_was_fmt_error = False

        elif role == "user":
            # Pull text from inside <information> tags for signal
            info_text = " ".join(_INFO_RE.findall(content))
            check = info_text or content

            if _FMT_ERROR_RE.search(check):
                fmt_errors += 1
                prev_was_fmt_error = True
            elif _TIMEOUT_RE.search(check):
                timeouts += 1
                prev_was_fmt_error = False
            elif _SERVER_ERROR_RE.search(check):
                server_errors += 1
                prev_was_fmt_error = False
            elif _KERNEL_RE.search(check):
                kernel_errors += 1
                prev_was_fmt_error = False
            elif _TRACEBACK_RE.search(check):
                runtime_errors += 1
                prev_was_fmt_error = False
            else:
                prev_was_fmt_error = False

    recovery_rate = recoveries / fmt_error_chances if fmt_error_chances else None
    return {
        "n_turns":        asst_seen,
        "fmt_errors":     fmt_errors,
        "timeouts":       timeouts,
        "server_errors":  server_errors,
        "kernel_errors":  kernel_errors,
        "runtime_errors": runtime_errors,
        "truncations":    truncations,
        "empty_resp":     empty_responses,
        "first_code":     first_turn_code,
        "first_dag":      first_turn_dag,
        "recovery_rate":  recovery_rate,
    }


@st.cache_data(show_spinner=False)
def load_failure_analysis_records() -> list:
    """Load all per-task JSON files, extract outcome + trajectory error stats.
    Returns one dict per (model, comp) — all modes included."""
    results: list[dict] = []
    seen:    set[tuple] = set()
    slug_re = re.compile(r"^(.+?)_(\d+)_(.+)$")

    def _slug_to_name(raw: str) -> str:
        name = raw.replace("_", "-")
        name = re.sub(r"\bgpt-(\d+)-(\d+)\b", r"gpt-\1.\2", name)
        return name

    for fpath in sorted(RESULTS_DIR.iterdir()):
        if fpath.suffix != ".json":
            continue
        m = slug_re.match(fpath.stem)
        if not m:
            continue
        comp, slug = m.group(1), m.group(3)

        mode      = None
        slug_base = slug
        for candidate in ["dags_adaptive-revision", "dags", "nodags"]:
            if slug.endswith(f"_{candidate}"):
                mode      = candidate
                slug_base = slug[: -len(f"_{candidate}")]
                break
        if mode is None:
            continue

        base_model = _slug_to_name(slug_base)
        suffix     = {"dags_adaptive-revision": " (adaptive)",
                      "nodags": " (nodags)", "dags": " (dags)"}[mode]
        model_full = f"{base_model}{suffix}"

        key = (model_full, comp)
        if key in seen:
            continue
        seen.add(key)

        try:
            item = json.load(open(fpath))
            if not isinstance(item, dict):
                continue
            traj = item.get("trajectory", [])
            if not traj:
                continue
            status = ((item.get("metrics") or {}).get("kaggle_submission") or {}) \
                         .get("details", {}).get("status", "") or ""
            outcome = ("complete" if "COMPLETE" in status else
                       "error"    if "ERROR"    in status else "no_submit")
            stats = _analyze_trajectory(traj)
            results.append({
                "model":      model_full,
                "base_model": base_model,
                "mode":       mode,
                "comp":       comp,
                "outcome":    outcome,
                "trajectory": traj,
                **stats,
            })
        except Exception:
            continue

    return results


def extract_traj_dags(all_data):
    """Build {(model, challenge): [dag_dicts]} from both results files and per-task files."""
    traj_dags = {}

    # Source 1: bundled results files
    for (model, mode), data in all_data.items():
        if mode not in ("dags", "dags_adaptive-revision"): continue
        for row in data:
            dags = _extract_embedded_dags(row.get("trajectory", []))
            ch   = row.get("challenge") or ""
            if dags and ch:
                traj_dags[(model, ch)] = dags

    # Source 2: per-task DAG files (covers partial runs / adaptive models)
    slug_re = re.compile(r'^(.+?)_(\d+)_(.+)$')
    for model_name, cfg in DAG_MODEL_CONFIGS.items():
        suffix = cfg["suffix"]
        for fpath in sorted(RESULTS_DIR.iterdir()):
            if not fpath.name.endswith(suffix): continue
            m = slug_re.match(fpath.stem)
            if not m: continue
            comp = m.group(1)          # e.g. "ashrae-energy-prediction"
            key  = (model_name, comp)
            if key in traj_dags: continue   # already populated from results file
            try:
                item = json.load(open(fpath))
            except Exception:
                continue
            if not isinstance(item, dict): continue
            traj = item.get("trajectory", [])
            if not traj: continue
            dags = _extract_embedded_dags(traj)
            if dags:
                traj_dags[key] = dags

    return traj_dags


def _extract_embedded_dags(trajectory):
    """Extract DAG dicts from assistant turns.

    Handles:
      1. <reasoning_dag>/<implementation_dag> XML tags  (DeepSeek adaptive)
      2. Closed fence with dag_type field               (GPT, Claude)
      3. Unclosed fence                                 (DeepSeek standard)
      4. Bare JSON at start of turn                     (older runs)
    """
    dags = []
    seen = set()

    def _add(obj, forced_dag_type=None):
        if not isinstance(obj, dict) or "nodes" not in obj:
            return
        if forced_dag_type:
            obj["dag_type"] = forced_dag_type
        if "dag_type" not in obj:
            return
        key = json.dumps(obj, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        dags.append({"turn": obj.get("_turn"), "dag_type": obj["dag_type"],
                     "nodes": obj.get("nodes", []), "edges": obj.get("edges", []),
                     "graph_id": obj.get("graph_id", "")})

    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        if not content:
            continue
        t = turn.get("turn")

        # 1. XML tag format: <reasoning_dag> / <implementation_dag>
        #    Handles both closed (</reasoning_dag>) and unclosed tags (DeepSeek adaptive)
        for tag, dag_type in [("reasoning_dag", "reasoning"), ("implementation_dag", "implementation")]:
            # Try closed tag first
            closed_matches = list(re.finditer(rf"<{tag}>([\s\S]*?)</{tag}>", content))
            for m in closed_matches:
                inner = m.group(1).strip()
                fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", inner)
                if fence_m:
                    inner = fence_m.group(1)
                for candidate in _find_json_objects(inner):
                    try:
                        obj = json.loads(candidate); obj["_turn"] = t
                        _add(obj, forced_dag_type=dag_type)
                    except Exception:
                        pass
            # If no closed match, try unclosed tag (DeepSeek adaptive emits <reasoning_dag> without closing)
            if not closed_matches:
                m_open = re.search(rf"<{tag}>([\s\S]+)", content)
                if m_open:
                    inner = m_open.group(1).strip()
                    fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", inner)
                    if fence_m:
                        inner = fence_m.group(1)
                    # Also try unclosed fence within unclosed tag
                    elif re.search(r"```(?:json)?", inner):
                        inner = re.sub(r"```(?:json)?\s*", "", inner).strip()
                    for candidate in _find_json_objects(inner):
                        try:
                            obj = json.loads(candidate); obj["_turn"] = t
                            _add(obj, forced_dag_type=dag_type)
                        except Exception:
                            pass

        # 2. Closed fences (only if no XML dag tags present — avoid double-counting)
        if not re.search(r"<(?:reasoning|implementation)_dag>", content):
            for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", content):
                for candidate in _find_json_objects(block):
                    try:
                        obj = json.loads(candidate); obj["_turn"] = t
                        _add(obj)
                    except Exception:
                        pass

            # 3. Unclosed fence (DeepSeek standard)
            m = re.search(r"```(?:json)?\s*([\s\S]+)$", content)
            if m and "```" not in m.group(1):
                for candidate in _find_json_objects(m.group(1)):
                    try:
                        obj = json.loads(candidate); obj["_turn"] = t
                        _add(obj)
                    except Exception:
                        pass

            # 4. Bare JSON (no fence at all)
            if not re.search(r"```", content):
                for candidate in _find_json_objects(content):
                    try:
                        obj = json.loads(candidate); obj["_turn"] = t
                        _add(obj)
                    except Exception:
                        pass


    return dags


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })


def draw_dag_on_ax(dag, ax, title=""):
    G = nx.DiGraph()
    node_labels = {}
    for n in dag.get("nodes",[]):
        nid = n["id"]; label = n.get("label", nid)
        if len(label) > 70: label = label[:67]+"..."
        G.add_node(nid); node_labels[nid] = label
    for e in dag.get("edges",[]): G.add_edge(e["source"], e["target"])
    if not G.nodes:
        ax.text(0.5,0.5,"(empty)",ha="center",va="center",transform=ax.transAxes)
        ax.set_title(title); ax.axis("off"); return
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    dtype = dag.get("dag_type","")
    nc = "#90CAF9" if dtype=="reasoning" else ("#A5D6A7" if dtype=="implementation" else "#FFCC80")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="->", arrowsize=15,
                           edge_color="#555", width=1.5, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=700, alpha=0.9)
    wrapped = {n:"\n".join(textwrap.wrap(lbl,22)) for n,lbl in node_labels.items() if n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=wrapped, ax=ax, font_size=6)
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.axis("off")


def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary helper
# ──────────────────────────────────────────────────────────────────────────────
def dp_summary_df(data_dict):
    rows = []
    for (model, mode), data in data_dict.items():
        sub      = [r for r in data if "COMPLETE" in str(r.get("status",""))]
        pcts     = [r["pub_percentile"]  for r in sub if r["pub_percentile"]  is not None]
        prpcts   = [r["priv_percentile"] for r in sub if r["priv_percentile"] is not None]
        above    = [r["pub_above_med"]   for r in sub if r["pub_above_med"]   is not None]
        times    = [r["execution_time"]  for r in data if r["execution_time"] is not None]
        turns    = [r["total_turns"]     for r in data if r["total_turns"]    is not None]
        medals   = [r for r in sub if r.get("public_medal") not in (None,"none")]
        tot_toks = [r["total_tokens"]    for r in data if r.get("total_tokens") is not None]
        rows.append({
            "Model":              MODEL_LABELS.get(model, model),
            "Mode":               mode,
            "N":                  len(data),
            "Coverage (%)":       round(100*len(sub)/len(data),1) if data else 0,
            "Medals":             len(medals),
            "Above Median (%)":   round(100*sum(above)/len(above),1) if above else 0,
            "Avg Pub Percentile": round(np.mean(pcts),2)   if pcts   else None,
            "Avg Priv Percentile":round(np.mean(prpcts),2) if prpcts else None,
            "Median Pub Pct":     round(np.median(pcts),2) if pcts   else None,
            "Avg Time (min)":     round(np.mean(times)/60,1) if times else None,
            "Avg Turns":          round(np.mean(turns),1)    if turns else None,
            "Avg Total Tokens":   round(np.mean(tot_toks))   if tot_toks else None,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ──────────────────────────────────────────────────────────────────────────────
def page_overview(dp_data, dp_hard_data, bio_data):
    st.header("📋 Overview — All Runs")
    all_models = sorted({m for (m,_) in list(dp_data)+list(dp_hard_data)+list(bio_data)})
    inv = {v:k for k,v in MODEL_LABELS.items()}
    sel_labels = st.multiselect("Filter models",
        [MODEL_LABELS.get(m,m) for m in all_models],
        default=[MODEL_LABELS.get(m,m) for m in all_models])
    sel_models = {inv.get(l,l) for l in sel_labels}

    st.subheader("DSPredict — Sampled")
    df_s = dp_summary_df({k:v for k,v in dp_data.items() if k[0] in sel_models})
    st.dataframe(df_s.set_index(["Model","Mode"]) if not df_s.empty else pd.DataFrame(), use_container_width=True)

    st.subheader("DSPredict — Hard")
    df_h = dp_summary_df({k:v for k,v in dp_hard_data.items() if k[0] in sel_models})
    st.dataframe(df_h.set_index(["Model","Mode"]) if not df_h.empty else pd.DataFrame(), use_container_width=True)

    st.subheader("DSBio")
    bio_rows = []
    for (model,mode),data in bio_data.items():
        if model not in sel_models: continue
        em    = [r["exact_match"]    for r in data if r["exact_match"]    is not None]
        times = [r["execution_time"] for r in data if r["execution_time"] is not None]
        turns = [r["total_turns"]    for r in data if r["total_turns"]    is not None]
        succ  = [r["success"]        for r in data if r["success"]        is not None]
        bio_rows.append({"Model":MODEL_LABELS.get(model,model),"Mode":mode,"N":len(data),
            "Exact Match (%)":  round(100*np.mean(em),1)          if em   else None,
            "Success Rate (%)": round(100*sum(succ)/len(succ),1)  if succ else None,
            "Avg Time (s)":     round(np.mean(times),1)            if times else None,
            "Avg Turns":        round(np.mean(turns),1)            if turns else None})
    if bio_rows:
        st.dataframe(pd.DataFrame(bio_rows).set_index(["Model","Mode"]), use_container_width=True)
    else:
        st.info("No DSBio data loaded.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DSPredict
# ──────────────────────────────────────────────────────────────────────────────
def page_dspredict(dp_data, dp_hard_data):
    st.header("🏆 DSPredict Analysis")
    apply_style()

    tab_sampled, tab_hard, tab_medals, tab_tokens, tab_challenge = st.tabs(
        ["Sampled","Hard","Medal Breakdown","Token Usage","Per-Challenge"])

    inv = {v:k for k,v in MODEL_LABELS.items()}

    def _cov(data):
        sub=[r for r in data if "COMPLETE" in str(r.get("status",""))]
        return 100*len(sub)/max(1,len(data))
    def _avg_pub(data):
        v=[r["pub_percentile"] for r in data if r.get("pub_percentile") is not None]
        return np.mean(v) * _cov(data) / 100  if v else 0
    def _avg_priv(data):
        v=[r["priv_percentile"] for r in data if r.get("priv_percentile") is not None]
        return np.mean(v) * _cov(data) / 100 if v else 0
    def _above(data):
        v=[r["pub_above_med"] for r in data if r.get("pub_above_med") is not None]
        return 100*sum(v)/N_PROBLEMS if v else 0
    def _time(data):
        v=[r["execution_time"]/60 for r in data if r.get("execution_time") is not None]
        return np.mean(v) if v else 0
    def _turns(data):
        v=[r["total_turns"] for r in data if r.get("total_turns") is not None]
        return np.mean(v) if v else 0

    # ── Sampled ──
    with tab_sampled:
        sampled_models = sorted({m for (m,_) in dp_data})
        sel = st.multiselect("Models",[MODEL_LABELS.get(m,m) for m in sampled_models],
            default=[MODEL_LABELS.get(m,m) for m in sampled_models], key="dp_samp_sel")
        sel_m = [inv.get(l,l) for l in sel]
        if not sel_m: st.info("Select at least one model."); return

        x = np.arange(len(sel_m)); w = 0.35

        # ── Core metrics grid ────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(28, 18))
        fig.suptitle("DSPredict-Sampled: DAG vs No-DAG", fontsize=16, fontweight="bold")
        metrics = [
            (_cov,      "Coverage (%)",    "Submission Coverage",    (0, 110)),
            (_avg_pub,  "Avg Percentile",  "Avg Normalized PUBLIC Percentile",   None),
            (_above,    "Rate (%)",        "Above Median Rate (%)",  (0, 110)),
            (_avg_priv, "Avg Percentile",  "Avg Normalized PRIVATE Percentile",  None),
            (_time,     "Minutes",         "Avg Execution Time",      None),
            (_turns,    "Turns",           "Avg Total Turns",         None),
        ]
        for ax, (fn, ylabel, title, ylim) in zip(axes.flat, metrics):
            vd = [fn(dp_data.get((m, "dags"),   [])) for m in sel_m]
            vn = [fn(dp_data.get((m, "nodags"), [])) for m in sel_m]
            b1 = ax.bar(x - w/2, vd, w, label="DAGs",    color=COLORS["dags"])
            b2 = ax.bar(x + w/2, vn, w, label="No-DAGs", color=COLORS["nodags"])
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in sel_m], rotation=45, ha="right")
            ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
            if ylim: ax.set_ylim(ylim)
            for bar in [*b1, *b2]:
                h = bar.get_height()
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=10)
        plt.tight_layout(); fig_to_st(fig)

        # ── Above-Median detailed breakdown ─────────────────────────────────
        st.subheader("📊 Above-Median Rate — Per Model × Mode")
        st.caption("Fraction of COMPLETE submissions that scored above the competition median. DAG vs No-DAG side by side.")
        above_rows = []
        for m in sel_m:
            for mode in ["dags", "nodags"]:
                data = dp_data.get((m, mode), [])
                sub  = [r for r in data if "COMPLETE" in str(r.get("status", ""))]
                ab   = [r["pub_above_med"] for r in sub if r.get("pub_above_med") is not None]
                above_rows.append({
                    "model": m, "mode": mode,
                    "label": f"{MODEL_LABELS.get(m, m)}\n({mode})",
                    "above_pct": 100 * sum(ab) / len(ab) if ab else None,
                    "n_submitted": len(sub), "n_above": sum(ab) if ab else 0,
                })
        df_above = pd.DataFrame(above_rows).dropna(subset=["above_pct"])
        if not df_above.empty:
            fig, ax = plt.subplots(figsize=(max(16, len(df_above) * 1.8), 10))
            bar_colors = [COLORS.get(r["mode"], "#888") for _, r in df_above.iterrows()]
            bars = ax.bar(range(len(df_above)), df_above["above_pct"],
                          color=bar_colors, edgecolor="white", width=0.65)
            ax.set_xticks(range(len(df_above)))
            ax.set_xticklabels(df_above["label"].tolist(), rotation=45, ha="right")
            ax.set_ylabel("Above Median Rate (%)"); ax.set_ylim(0, 110)
            ax.set_title("Above-Median Rate by Model and Mode (DSPredict Sampled)")
            ax.axhline(50, color="red", ls="--", lw=1.5, alpha=0.5, label="50% baseline")
            for i, (bar, (_, row)) in enumerate(zip(bars, df_above.iterrows())):
                h = bar.get_height()
                ax.text(i, h + 1.5, f"{h:.1f}%\n({int(row['n_above'])}/{int(row['n_submitted'])})",
                        ha="center", va="bottom", fontsize=10)
            handles = [mpatches.Patch(facecolor=COLORS["dags"], label="DAGs"),
                       mpatches.Patch(facecolor=COLORS["nodags"], label="No-DAGs"),
                       plt.Line2D([0], [0], color="red", ls="--", label="50% baseline")]
            ax.legend(handles=handles)
            plt.tight_layout(); fig_to_st(fig)
        else:
            st.info("No above-median data available for selected models.")

        # ── Submission status breakdown ──────────────────────────────────────
        st.subheader("📋 Submission Status Breakdown")
        st.caption("COMPLETE = successful Kaggle submission · ERROR = runtime/submission error · NO_SUBMISSION = agent didn't produce a submission")
        STATUS_COLORS = {
            "SUBMISSIONSTATUS.COMPLETE":      "#4CAF50",
            "SUBMISSIONSTATUS.ERROR":         "#F44336",
            "SUBMISSIONSTATUS.NO_SUBMISSION": "#9E9E9E",
        }
        STATUS_SHORT = {
            "SUBMISSIONSTATUS.COMPLETE":      "Complete",
            "SUBMISSIONSTATUS.ERROR":         "Error",
            "SUBMISSIONSTATUS.NO_SUBMISSION": "No Submit",
        }
        status_data = []
        for m in sel_m:
            for mode in ["dags", "nodags"]:
                data = dp_data.get((m, mode), [])
                if not data: continue
                counts: dict = {}
                for r in data:
                    s = r.get("status", "") or ""
                    if not s or "COMPLETE" not in s and "ERROR" not in s:
                        s = "SUBMISSIONSTATUS.NO_SUBMISSION"
                    elif "ERROR" in s:
                        s = "SUBMISSIONSTATUS.ERROR"
                    else:
                        s = "SUBMISSIONSTATUS.COMPLETE"
                    counts[s] = counts.get(s, 0) + 1
                status_data.append({
                    "model": m, "mode": mode,
                    "label": f"{MODEL_LABELS.get(m, m)}\n({mode})",
                    "total": len(data), **counts,
                })
        if status_data:
            df_st = pd.DataFrame(status_data).fillna(0)
            all_statuses = ["SUBMISSIONSTATUS.COMPLETE", "SUBMISSIONSTATUS.ERROR", "SUBMISSIONSTATUS.NO_SUBMISSION"]
            for s in all_statuses:
                if s not in df_st.columns: df_st[s] = 0

            fig, ax = plt.subplots(figsize=(max(16, len(df_st) * 1.8), 10))
            bottoms = np.zeros(len(df_st))
            for status in all_statuses:
                vals = df_st[status].values
                bars = ax.bar(range(len(df_st)), vals, bottom=bottoms,
                              color=STATUS_COLORS[status], label=STATUS_SHORT[status],
                              edgecolor="white", width=0.65)
                for i, (bar, v) in enumerate(zip(bars, vals)):
                    if v >= 1:
                        ax.text(i, bottoms[i] + v / 2, f"{int(v)}",
                                ha="center", va="center", fontsize=11, fontweight="bold", color="white")
                bottoms += vals
            ax.set_xticks(range(len(df_st)))
            ax.set_xticklabels(df_st["label"].tolist(), rotation=45, ha="right")
            ax.set_ylabel("Number of runs"); ax.set_title("Submission Status Breakdown (DSPredict Sampled)")
            ax.legend(loc="upper right")
            plt.tight_layout(); fig_to_st(fig)

            # Table view
            display_df = df_st[["label", "total"] + all_statuses].copy()
            display_df.columns = ["Model (Mode)", "Total"] + [STATUS_SHORT[s] for s in all_statuses]
            st.dataframe(display_df.set_index("Model (Mode)"), use_container_width=True)

        st.subheader("Public vs Private Percentile (per submission)")
        sc_m = inv.get(st.selectbox("Model", [MODEL_LABELS.get(m, m) for m in sel_m], key="scat_m"), sel_m[0])
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        for mode, color in [("dags", COLORS["dags"]), ("nodags", COLORS["nodags"])]:
            data = dp_data.get((sc_m, mode), [])
            pub = [r["pub_percentile"]  for r in data if r.get("pub_percentile") and r.get("priv_percentile")]
            prv = [r["priv_percentile"] for r in data if r.get("pub_percentile") and r.get("priv_percentile")]
            ax2.scatter(pub, prv, alpha=0.7, color=color, label=mode, s=80)
        ax2.plot([0, 100], [0, 100], "k--", alpha=0.3, lw=1)
        ax2.set_xlabel("Public Percentile"); ax2.set_ylabel("Private Percentile")
        ax2.set_title(f"{MODEL_LABELS.get(sc_m, sc_m)}: Public vs Private"); ax2.legend()
        fig_to_st(fig2)

    # ── Hard ──
    with tab_hard:
        hard_keys = list(dp_hard_data.keys())
        if not hard_keys: st.info("No hard data loaded."); return
        all_hm = sorted({m for (m,_) in hard_keys})
        sel_h  = st.multiselect("Models",[MODEL_LABELS.get(m,m) for m in all_hm],
            default=[MODEL_LABELS.get(m,m) for m in all_hm], key="dp_hard_sel")
        sel_hm = [inv.get(l,l) for l in sel_h]
        filtered = {k:v for k,v in dp_hard_data.items() if k[0] in sel_hm}
        rk   = list(filtered.keys())
        rlab = [f"{MODEL_LABELS.get(m,m)}\n({md})" for m,md in rk]
        xp   = np.arange(len(rk)); bw = max(0.4, min(0.8, 5/max(1,len(rk))))

        # Core metrics
        fig, axes = plt.subplots(2, 2, figsize=(max(22, len(rk)*2.5), 18))
        fig.suptitle("DSPredict-Hard: Key Metrics", fontsize=16, fontweight="bold")
        for ax, (fn, ylabel, title) in zip(axes.flat, [
            (_cov,      "Coverage (%)",         "Submission Coverage"),
            (_avg_pub,  "Avg Pub Percentile",   "Avg PUBLIC Percentile"),
            (_avg_priv, "Avg Priv Percentile",  "Avg PRIVATE Percentile"),
            (_above,    "Above Median Rate (%)", "Above Median Rate"),
        ]):
            heights = [fn(filtered.get(k, [])) or 0 for k in rk]
            bc = [COLORS.get(md, "#9C27B0") for _,md in rk]
            bars = ax.bar(xp, heights, width=bw, color=bc, edgecolor="white")
            ax.set_xticks(xp); ax.set_xticklabels(rlab, rotation=45, ha="right")
            ax.set_ylabel(ylabel); ax.set_title(title)
            for bar, h in zip(bars, heights):
                if h > 0.5:
                    ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=10)
        handles = [mpatches.Patch(facecolor=COLORS.get(md, "#9C27B0"), label=md)
                   for md in ["dags", "nodags", "dags_adaptive-revision"]]
        fig.legend(handles=handles, loc="upper right")
        plt.tight_layout(); fig_to_st(fig)

        # ── Status breakdown for Hard ────────────────────────────────────────
        st.subheader("📋 Submission Status Breakdown (Hard)")
        hard_status = []
        for (m, mode), data in filtered.items():
            if not data: continue
            counts: dict = {}
            for r in data:
                s = r.get("status", "") or ""
                if "ERROR" in s:         s = "SUBMISSIONSTATUS.ERROR"
                elif "COMPLETE" in s:    s = "SUBMISSIONSTATUS.COMPLETE"
                else:                    s = "SUBMISSIONSTATUS.NO_SUBMISSION"
                counts[s] = counts.get(s, 0) + 1
            hard_status.append({
                "label": f"{MODEL_LABELS.get(m,m)}\n({mode})",
                "total": len(data), **counts,
            })
        if hard_status:
            df_hs = pd.DataFrame(hard_status).fillna(0)
            for s in ["SUBMISSIONSTATUS.COMPLETE","SUBMISSIONSTATUS.ERROR","SUBMISSIONSTATUS.NO_SUBMISSION"]:
                if s not in df_hs.columns: df_hs[s] = 0
            fig, ax = plt.subplots(figsize=(max(16, len(df_hs)*2), 10))
            bottoms = np.zeros(len(df_hs))
            for status in ["SUBMISSIONSTATUS.COMPLETE","SUBMISSIONSTATUS.ERROR","SUBMISSIONSTATUS.NO_SUBMISSION"]:
                vals = df_hs[status].values
                bars = ax.bar(range(len(df_hs)), vals, bottom=bottoms,
                              color=STATUS_COLORS[status], label=STATUS_SHORT[status],
                              edgecolor="white", width=0.65)
                for i, (bar, v) in enumerate(zip(bars, vals)):
                    if v >= 1:
                        ax.text(i, bottoms[i]+v/2, f"{int(v)}", ha="center", va="center",
                                fontsize=11, fontweight="bold", color="white")
                bottoms += vals
            ax.set_xticks(range(len(df_hs)))
            ax.set_xticklabels(df_hs["label"].tolist(), rotation=45, ha="right")
            ax.set_ylabel("Number of runs"); ax.set_title("Submission Status Breakdown (DSPredict Hard)")
            ax.legend(loc="upper right")
            plt.tight_layout(); fig_to_st(fig)

    # ── Medal Breakdown ──
    with tab_medals:
        st.subheader("Medal Type Breakdown — All Runs")
        all_rk    = list(dp_data.keys())+[k for k in dp_hard_data if k not in dp_data]
        all_rd    = {**dp_data,**dp_hard_data}
        rlab_full = [f"{MODEL_LABELS.get(m,m)}\n{md}" for m,md in all_rk]
        xp=np.arange(len(all_rk)); bw=max(0.4,min(0.8,6/max(1,len(all_rk))))

        fig,axes=plt.subplots(2,2,figsize=(max(28, len(all_rk)*2.8), 20))
        fig.suptitle("DSPredict: Medal Type Breakdown — Public vs Private LB",fontsize=16,fontweight="bold")
        for row_i,(lb_key,lb_label) in enumerate([("public_medal","Public LB"),("private_medal","Private LB")]):
            ax=axes[row_i,0]; bottoms=np.zeros(len(all_rk))
            for mtype in MEDAL_TYPES:
                heights=[sum(1 for r in [r for r in all_rd.get(k,[]) if "COMPLETE" in str(r.get("status",""))]
                             if r.get(lb_key)==mtype) for k in all_rk]
                bars=ax.bar(xp,heights,bottom=bottoms,width=bw,color=MEDAL_COLORS[mtype],
                            label=mtype.capitalize(),edgecolor="white")
                for bar,h in zip(bars,heights):
                    if h>0: ax.text(bar.get_x()+bar.get_width()/2,bar.get_y()+h/2,str(int(h)),
                                    ha="center",va="center",fontsize=12,fontweight="bold")
                bottoms+=np.array(heights)
            ax.set_xticks(xp); ax.set_xticklabels(rlab_full,fontsize=11,rotation=45, ha="right")
            ax.set_ylabel("Number of Medals"); ax.set_title(f"Medal Count — {lb_label}")
            ax.legend(loc="upper right")

            ax=axes[row_i,1]; bottoms_pct=np.zeros(len(all_rk))
            for mtype in MEDAL_TYPES_FULL:
                heights_pct=[]
                for k in all_rk:
                    sub=[r for r in all_rd.get(k,[]) if "COMPLETE" in str(r.get("status",""))]
                    heights_pct.append(100*sum(1 for r in sub if (r.get(lb_key) or "none")==mtype)/max(1,len(sub)))
                bars=ax.bar(xp,heights_pct,bottom=bottoms_pct,width=bw,
                            color=MEDAL_COLORS[mtype],label=MEDAL_LABELS_FULL[mtype],edgecolor="white")
                for bar,h in zip(bars,heights_pct):
                    if h>4: ax.text(bar.get_x()+bar.get_width()/2,bar.get_y()+h/2,
                                    f"{h:.0f}%",ha="center",va="center",fontsize=11,fontweight="bold")
                bottoms_pct+=np.array(heights_pct)
            ax.set_xticks(xp); ax.set_xticklabels(rlab_full,fontsize=11,rotation=45, ha="right")
            ax.set_ylabel("% of Submitted Tasks"); ax.set_title(f"Medal Share — {lb_label}")
            ax.legend(loc="upper right"); ax.set_ylim(0,108)
            for xi,k in enumerate(all_rk):
                sub=[r for r in all_rd.get(k,[]) if "COMPLETE" in str(r.get("status",""))]
                ax.text(xi,-8,f"n={len(sub)}",ha="center",va="top",fontsize=10,color="dimgray")
        plt.tight_layout(); fig_to_st(fig)

    # ── Tokens ──
    with tab_tokens:
        st.subheader("Token Usage per Run")
        all_rk2  = list(dp_data.keys())+list(dp_hard_data.keys())
        all_rd2  = {**dp_data,**dp_hard_data}
        tok_lab  = [f"{MODEL_LABELS.get(m,m)}\n{md}" for m,md in all_rk2]
        xp2=np.arange(len(all_rk2))
        def _avg_tok(field, dl):
            v=[r[field] for r in dl if r.get(field) is not None]
            return np.mean(v)/1000 if v else 0
        fig,axes=plt.subplots(1,3,figsize=(max(28, len(all_rk2)*2.5), 12))
        fig.suptitle("Token Usage per Run",fontsize=16,fontweight="bold")
        for ax,(field,ylabel) in zip(axes,[("input_tokens","Input Tokens (K)"),
                                           ("output_tokens","Output Tokens (K)"),
                                           ("total_tokens","Total Tokens (K)")]):
            heights=[_avg_tok(field, (dp_data.get(k) or dp_hard_data.get(k,[]))) for k in all_rk2]
            bc=[COLORS.get(md,"#9C27B0") for _,md in all_rk2]
            bars=ax.bar(xp2,heights,color=bc,edgecolor="white")
            ax.set_xticks(xp2); ax.set_xticklabels(tok_lab,rotation=45, ha="right",fontsize=10)
            ax.set_ylabel(ylabel); ax.set_title(ylabel)
            for bar,v in zip(bars,heights):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,v+0.3,f"{v:.1f}K",ha="center",va="bottom",fontsize=10)
        handles=[mpatches.Patch(facecolor=COLORS.get(md,"#9C27B0"),label=md)
                 for md in ["dags","nodags","dags_adaptive-revision"]]
        fig.legend(handles=handles,loc="upper right")
        plt.tight_layout(); fig_to_st(fig)

    # ── Per-Challenge ──
    with tab_challenge:
        st.subheader("Per-Challenge Results")
        ds  = st.radio("Dataset",["Sampled","Hard"],horizontal=True,key="ch_ds")
        met = st.radio("Metric",["pub_percentile","priv_percentile","public_medal"],
            format_func=lambda x:{"pub_percentile":"Public Percentile",
                                   "priv_percentile":"Private Percentile","public_medal":"Medal"}[x],
            horizontal=True,key="ch_met")
        src = dp_data if ds=="Sampled" else dp_hard_data
        dfs=[]
        for (model,mode),data in src.items():
            df=pd.DataFrame(data).drop(columns=["trajectory"],errors="ignore")
            df["model"]=MODEL_LABELS.get(model,model); df["mode"]=mode; dfs.append(df)
        if dfs:
            combined=pd.concat(dfs,ignore_index=True)
            pivot=combined.pivot_table(index="challenge",columns=["model","mode"],values=met,aggfunc="first")
            if met!="public_medal": pivot=pivot.round(2)
            st.dataframe(pivot,use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DSBio
# ──────────────────────────────────────────────────────────────────────────────
def page_dsbio(bio_data):
    st.header("🧬 DSBio Analysis")
    apply_style()
    if not bio_data: st.info("No DSBio data loaded."); return

    bio_models = sorted({m for (m,_) in bio_data})
    inv = {v:k for k,v in MODEL_LABELS.items()}
    sel = st.multiselect("Models",[MODEL_LABELS.get(m,m) for m in bio_models],
        default=[MODEL_LABELS.get(m,m) for m in bio_models],key="bio_sel")
    sel_m = [inv.get(l,l) for l in sel]
    if not sel_m: return

    x=np.arange(len(sel_m)); w=0.35
    def bio_mean(m,md,key,scale=1.0):
        data=bio_data.get((m,md),[])
        v=[r[key] for r in data if r.get(key) is not None]
        return scale*np.mean(v) if v else 0

    fig,axes=plt.subplots(1,3,figsize=(20, 8))
    fig.suptitle("DSBio: DAG vs No-DAG",fontsize=14,fontweight="bold")
    for ax,(key,scale,ylabel,title) in zip(axes,[
        ("exact_match",100,"Exact Match (%)","Exact Match Rate"),
        ("execution_time",1/60,"Minutes","Avg Execution Time"),
        ("total_turns",1,"Turns","Avg Total Turns"),
    ]):
        vd=[bio_mean(m,"dags",key,scale)   for m in sel_m]
        vn=[bio_mean(m,"nodags",key,scale) for m in sel_m]
        b1=ax.bar(x-w/2,vd,w,label="DAGs",   color=COLORS["dags"])
        b2=ax.bar(x+w/2,vn,w,label="No-DAGs",color=COLORS["nodags"])
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS.get(m,m) for m in sel_m],rotation=90, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        if key=="exact_match": ax.set_ylim(0,110)
        for bar in [*b1,*b2]:
            h=bar.get_height()
            if h>0: ax.text(bar.get_x()+bar.get_width()/2,h+0.5,f"{h:.1f}",ha="center",va="bottom",fontsize=9)
    plt.tight_layout(); fig_to_st(fig)

    st.subheader("Per-sample breakdown")
    m_ch=st.selectbox("Model",sel,key="bio_model_ch")
    md_ch=st.selectbox("Mode",["dags","nodags"],key="bio_mode_ch")
    data=bio_data.get((inv.get(m_ch,m_ch),md_ch),[])
    if data:
        st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!="trajectory"} for r in data]),
                     use_container_width=True)
    else:
        st.info("No data for this selection.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DAG Visualizer
# ──────────────────────────────────────────────────────────────────────────────
def page_dag_visualizer(dp_data, dp_hard_data):
    st.header("🔀 DAG Visualizer")
    apply_style()

    with st.spinner("Extracting DAGs from trajectories…"):
        all_data = {**dp_data,**{k:v for k,v in dp_hard_data.items() if k not in dp_data}}
        traj_dags = extract_traj_dags(all_data)

    if not traj_dags:
        st.warning("No trajectory DAGs found in loaded runs."); return

    inv = {v:k for k,v in MODEL_LABELS.items()}
    models_with = sorted({m for m,_ in traj_dags})
    sel_model = st.selectbox("Model",[MODEL_LABELS.get(m,m) for m in models_with],key="viz_model")
    sel_m = inv.get(sel_model,sel_model)

    challenges = sorted({ch for m,ch in traj_dags if m==sel_m and ch})
    if not challenges: st.info("No challenges with DAGs for this model."); return
    sel_ch = st.selectbox("Challenge",challenges,key="viz_ch")

    dags = traj_dags.get((sel_m,sel_ch),[])
    if not dags: st.info("No DAGs found."); return

    r_dags = [d for d in dags if d["dag_type"]=="reasoning"]
    i_dags = [d for d in dags if "implementation" in d["dag_type"]]
    o_dags = [d for d in dags if d not in r_dags and d not in i_dags]

    st.markdown(f"**{len(dags)} total DAGs** — {len(r_dags)} reasoning · {len(i_dags)} implementation · {len(o_dags)} other")

    type_filter = st.multiselect("DAG types to show",["reasoning","implementation","other"],
        default=["reasoning","implementation","other"],key="viz_type")
    to_show=[]
    if "reasoning"      in type_filter: to_show+=[(d,"Reasoning")      for d in r_dags]
    if "implementation" in type_filter: to_show+=[(d,"Implementation") for d in i_dags]
    if "other"          in type_filter: to_show+=[(d,"Other")          for d in o_dags]
    if not to_show: return

    total = len(to_show)

    # ── All steps (grid) ──────────────────────────────────────────────────────
    st.subheader(f"All {total} steps")
    cols_per_row = st.slider("Columns per row", 1, 6, min(3, total), key="viz_cols")
    n_rows = (total + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(7 * cols_per_row, 6 * n_rows))
    # Normalise axes to a flat list regardless of shape
    if n_rows == 1 and cols_per_row == 1:
        flat_axes = [axes]
    elif n_rows == 1:
        flat_axes = list(axes)
    elif cols_per_row == 1:
        flat_axes = list(axes)
    else:
        flat_axes = [ax for row in axes for ax in row]

    for i, (d, dl) in enumerate(to_show):
        n_nodes = len(d.get("nodes", [])); n_edges = len(d.get("edges", []))
        title = f"Step {i+1}/{total} · {dl}\n{n_nodes} nodes · {n_edges} edges · turn {d.get('turn','?')}"
        draw_dag_on_ax(d, flat_axes[i], title)
    for i in range(total, len(flat_axes)):
        flat_axes[i].axis("off")

    fig.suptitle(f"{MODEL_LABELS.get(sel_m, sel_m)} · {sel_ch}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_to_st(fig)

    # ── Focused view with slider ──────────────────────────────────────────────
    st.subheader("Zoom in on a single step")
    idx = st.slider("Step", 1, total, 1, key="viz_step") - 1
    dag, dag_label = to_show[idx]
    n_nodes = len(dag.get("nodes",[])); n_edges = len(dag.get("edges",[]))
    st.caption(f"Step {idx+1}/{total} — {dag_label} — {n_nodes} nodes, {n_edges} edges — turn {dag.get('turn','?')}")

    fig2, ax2 = plt.subplots(figsize=(18, 10))
    draw_dag_on_ax(dag, ax2, f"{dag_label} DAG · {sel_ch} · step {idx+1}/{total}")
    plt.tight_layout(); fig_to_st(fig2)

    with st.expander("Nodes & Edges table"):
        c1,c2=st.columns(2)
        with c1:
            st.write("**Nodes**")
            st.dataframe(pd.DataFrame(dag.get("nodes",[])),use_container_width=True)
        with c2:
            st.write("**Edges**")
            st.dataframe(pd.DataFrame(dag.get("edges",[])),use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DAG Consistency
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Semantic embedding helpers
# ──────────────────────────────────────────────────────────────────────────────
import pickle

# ── Embedding config (mirrors dag_consistency_analysis.ipynb) ─────────────────
EMBEDDING_CONFIG = {
    "model":          "fireworks_ai/fireworks/qwen3-embedding-8b",
    "use_fallback":   False,   # set True to force sentence-transformers
    "fallback_model": "all-MiniLM-L6-v2",
}


@st.cache_resource(show_spinner=False)
def _get_embedder():
    """Return embedder: litellm (qwen3-embedding-8b) or sentence-transformers fallback."""
    if not EMBEDDING_CONFIG["use_fallback"]:
        try:
            import litellm, os
            # Quick probe — embed one token to confirm the API key + model work
            litellm.embedding(model=EMBEDDING_CONFIG["model"], input=["test"])
            return ("litellm", EMBEDDING_CONFIG["model"])
        except Exception as e:
            st.sidebar.warning(f"Fireworks embedding unavailable ({e}), falling back to sentence-transformers.")
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBEDDING_CONFIG["fallback_model"])
        return ("st", m)
    except Exception:
        return None


def _mean_embed(embedder_obj, texts):
    """Embed a list of texts, return mean unit-normalised vector, or None."""
    if not texts or not any(t for t in texts if t and t.strip()):
        return None
    if embedder_obj is None:
        return None
    kind = embedder_obj[0]
    try:
        if kind == "litellm":
            import litellm
            resp = litellm.embedding(model=embedder_obj[1], input=list(texts))
            vecs = np.array([d["embedding"] for d in resp.data], dtype=np.float32)
        else:
            vecs = embedder_obj[1].encode(list(texts), show_progress_bar=False,
                                          normalize_embeddings=False)
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).astype(np.float32) if norm > 0 else None
    except Exception:
        return None


def _cosine(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))   # already unit-normalised


_EMBED_CACHE_PATH = RESULTS_DIR / ".streamlit_embed_cache.pkl"


def _load_embed_cache():
    if _EMBED_CACHE_PATH.exists():
        try:
            with open(_EMBED_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_embed_cache(cache):
    try:
        with open(_EMBED_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


def _extract_traj_texts_impl(traj):
    """Return list of (reasoning_str, code_str, step, is_revision, after_error, reward)."""
    turns_out = []
    prev_had_error = False
    for turn in traj:
        role = turn.get("role", "")
        content = turn.get("content", "") or ""
        if role == "user":
            prev_had_error = bool(re.search(r"<error>", content))
            continue
        if role != "assistant":
            continue
        r_m = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
        p_m = re.search(r"<python>(.*?)</python>", content, re.DOTALL)
        if not p_m:
            p_m = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
        reasoning = r_m.group(1).strip() if r_m else ""
        code      = p_m.group(1).strip() if p_m else ""
        if not reasoning and not code and content.strip():
            if not re.search(r"<(information|error)>", content):
                reasoning = content.strip()
        if not reasoning and not code:
            prev_had_error = False
            continue
        is_rev = bool(re.search(r"PLAN REVISION|REPLAN|REPLANNING", reasoning, re.IGNORECASE))
        turns_out.append((reasoning, code, turn.get("turn", 0), is_rev, prev_had_error, turn.get("reward", 0.0)))
        prev_had_error = False
    return turns_out


async def _aembed_many(model_name: str, items: list[tuple], cache: dict, concurrency: int = 10):
    """Async-embed a batch of (key, texts) pairs, writing results into cache."""
    import asyncio, litellm

    sem = asyncio.Semaphore(concurrency)

    async def _one(key, texts):
        async with sem:
            try:
                resp = await litellm.aembedding(model=model_name, input=list(texts))
                vecs = np.array([d["embedding"] for d in resp.data], dtype=np.float32)
                mean = vecs.mean(axis=0)
                norm = np.linalg.norm(mean)
                cache[key] = (mean / norm).astype(np.float32) if norm > 0 else None
            except Exception:
                cache[key] = None

    await asyncio.gather(*[_one(k, t) for k, t in items])


def _run_async(coro):
    """Run a coroutine from sync context (safe inside Streamlit worker threads)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an already-running loop (shouldn't happen in cache threads,
            # but handle gracefully by creating a new loop in a thread)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _prefetch_embeddings(pending: list[tuple], embedder, cache: dict, label: str = ""):
    """
    Given a list of (key, texts) pairs, embed all cache misses.
    Uses async parallel API calls for litellm, sequential for sentence-transformers.
    Shows both an st.progress bar (UI) and tqdm (terminal).
    """
    from tqdm import tqdm

    missing = [
        (k, t) for k, t in pending
        if k not in cache or (cache[k] is None and any((x or "").strip() for x in t))
    ]
    if not missing or embedder is None:
        return

    n = len(missing)
    kind = embedder[0]
    prog = st.progress(0, text=f"{label}: embedding {n} text groups…")

    if kind == "litellm":
        chunk = 50
        with tqdm(total=n, desc=label, unit="grp") as pbar:
            for i in range(0, n, chunk):
                batch = missing[i:i + chunk]
                _run_async(_aembed_many(embedder[1], batch, cache))
                done = min(i + chunk, n)
                pbar.update(len(batch))
                prog.progress(done / n, text=f"{label}: {done}/{n} embedded…")
    else:
        for i, (key, texts) in enumerate(tqdm(missing, desc=label, unit="grp")):
            has_content = any((t or "").strip() for t in texts)
            cache[key] = _mean_embed(embedder, texts) if has_content else None
            if i % 10 == 0:
                prog.progress((i + 1) / n, text=f"{label}: {i+1}/{n} embedded…")

    _save_embed_cache(cache)
    prog.empty()


@st.cache_data(show_spinner=False, persist="disk")
def compute_semantic_df(records_fingerprint: str):
    """
    Compute semantic metrics from both DAG labels and raw trajectory text.
    Returns (df_sem, df_rev, df_intermodel).

    df_sem      — per-run DAG-based cosine similarities (only for runs with DAGs)
    df_rev      — per-run trajectory-text revision analysis (all runs)
    df_intermodel — pairwise cross-model cosine sims for the same competition
    """
    records  = load_dag_consistency_records()
    embedder = _get_embedder()
    cache    = _load_embed_cache()
    new_entries = 0

    # ── Pre-fetch all embeddings async before main loop ───────────────────────
    if embedder is not None:
        pending: list[tuple] = []
        for rec in records:
            m, c = rec["model"], rec["comp"]
            if rec["has_dag"]:
                r_lbls = rec.get("r_lbls", [])
                i_lbls = rec.get("i_lbls", [])
                all_code = rec.get("all_code", "")
                if r_lbls: pending.append((f"{m}|{c}|r", r_lbls))
                if i_lbls: pending.append((f"{m}|{c}|i", i_lbls))
                cc = [all_code[k:k+512] for k in range(0, min(len(all_code), 4096), 512)]
                if cc: pending.append((f"{m}|{c}|code", cc))
            for t_i, (reasoning, code, *_) in enumerate(_extract_traj_texts_impl(rec.get("trajectory", []))):
                texts = [reasoning] if reasoning else [code[:512]]
                if texts: pending.append((f"{m}|{c}|traj{t_i}", texts))
        _prefetch_embeddings(pending, embedder, cache, label="🧠 Semantic embeddings")

    def _emb(key, texts):
        nonlocal new_entries
        has_content = bool(texts and any((t or "").strip() for t in texts))
        if key in cache and (cache[key] is not None or not has_content):
            return cache[key]
        vec = _mean_embed(embedder, texts) if embedder and has_content else None
        cache[key] = vec
        new_entries += 1
        if new_entries % 20 == 0:
            _save_embed_cache(cache)
        return vec

    def _extract_traj_texts(traj):
        return _extract_traj_texts_impl(traj)

    sem_rows, rev_rows, intermodel_rows = [], [], []
    # {comp: [(model, final_emb), ...]}  for inter-model matrix
    comp_final_embs: dict[str, list] = defaultdict(list)

    for rec in records:
        model   = rec["model"]
        comp    = rec["comp"]
        success = rec["outcome"] == "complete"
        traj    = rec.get("trajectory", [])

        # ── DAG-based semantics (only when DAGs present) ──────────────────────
        if rec["has_dag"]:
            r_lbls   = rec.get("r_lbls", [])
            i_lbls   = rec.get("i_lbls", [])
            all_code = rec.get("all_code", "")

            emb_r    = _emb(f"{model}|{comp}|r", r_lbls)
            emb_i    = _emb(f"{model}|{comp}|i", i_lbls) if i_lbls else None
            cc = [all_code[k:k+512] for k in range(0, min(len(all_code), 4096), 512)]
            emb_code = _emb(f"{model}|{comp}|code", cc)

            r_i_sem  = _cosine(emb_r, emb_i)
            r_code   = _cosine(emb_r, emb_code)
            i_code   = _cosine(emb_i, emb_code) if emb_i is not None else None
            # For adaptive models (no impl dag): use r_code as best available code grounding
            code_sem = i_code if i_code is not None else r_code

            sem_rows.append({
                "model": model, "comp": comp, "success": success,
                "r_i_semantic":    r_i_sem,
                "r_code_semantic": r_code,
                "i_code_semantic": i_code,
                "code_semantic":   code_sem,   # i_code when available, else r_code
                "has_impl_dag":    emb_i is not None,
                "r_i_jaccard":     rec.get("ri_jaccard"),
                "code_cov":        rec.get("code_cov"),
            })

        # ── Trajectory text revision analysis (all records) ───────────────────
        if not traj:
            continue
        traj_turns = _extract_traj_texts(traj)
        if not traj_turns:
            continue

        # Embed each reasoning/code block
        turn_embs = []
        for t_i, (reasoning, code, *_) in enumerate(traj_turns):
            texts = [reasoning] if reasoning else [code[:512]]
            turn_embs.append(_emb(f"{model}|{comp}|traj{t_i}", texts))

        # Consecutive cosine similarities
        consec_sims, post_err_drifts, rev_sims = [], [], []
        for t_i in range(1, len(traj_turns)):
            sim = _cosine(turn_embs[t_i-1], turn_embs[t_i])
            if sim is None:
                continue
            consec_sims.append(sim)
            _, _, _, is_rev, after_err, _ = traj_turns[t_i]
            if after_err:
                post_err_drifts.append(1.0 - sim)
            if is_rev:
                rev_sims.append(sim)

        first_last = _cosine(turn_embs[0], turn_embs[-1]) if len(turn_embs) >= 2 else None

        # Store final embedding for inter-model comparison
        final_emb = next((e for e in reversed(turn_embs) if e is not None), None)
        if final_emb is not None:
            comp_final_embs[comp].append((model, final_emb))

        n_revs = sum(1 for _, _, _, is_rev, _, _ in traj_turns if is_rev)
        mc = float(np.mean(consec_sims)) if consec_sims else None
        rev_rows.append({
            "model":              model,
            "comp":               comp,
            "success":            success,
            "n_turns":            len(traj_turns),
            "n_revisions":        n_revs,
            "mean_consec_sim":    mc,
            "mean_consec_drift":  (1.0 - mc) if mc is not None else None,
            "post_error_drift":   float(np.mean(post_err_drifts)) if post_err_drifts else None,
            "revision_sim":       float(np.mean(rev_sims))       if rev_sims         else None,
            "first_last_sim":     first_last,
            "first_last_drift":   (1.0 - first_last) if first_last is not None else None,
        })

    # ── Inter-model pairwise similarities ─────────────────────────────────────
    for comp, model_embs in comp_final_embs.items():
        for i in range(len(model_embs)):
            for j in range(i+1, len(model_embs)):
                ma, ea = model_embs[i]
                mb, eb = model_embs[j]
                sim = _cosine(ea, eb)
                if sim is not None:
                    intermodel_rows.append({
                        "comp": comp, "model_a": ma, "model_b": mb,
                        "cos_sim": sim,
                        "pair": f"{ma}\nvs\n{mb}",
                    })

    if new_entries:
        _save_embed_cache(cache)

    return (
        pd.DataFrame(sem_rows),
        pd.DataFrame(rev_rows),
        pd.DataFrame(intermodel_rows),
    )


_SEMANTIC_CACHE_VERSION = "v6"   # bump when semantic computation logic changes

def _records_fingerprint(records):
    ids = [r.get("fname","") for r in records]
    return _SEMANTIC_CACHE_VERSION + "_" + hashlib.md5(json.dumps(ids).encode()).hexdigest()


def _drift_fingerprint():
    """Fingerprint based on all per-task files present (dags + nodags)."""
    fnames = sorted(
        f.name for f in RESULTS_DIR.iterdir()
        if f.suffix == ".json" and re.match(r"^.+?_\d+_.+_(dags|nodags)", f.stem)
    )
    return _SEMANTIC_CACHE_VERSION + "_drift_" + hashlib.md5(json.dumps(fnames).encode()).hexdigest()


@st.cache_data(show_spinner=False, persist="disk")
def compute_drift_df(fingerprint: str) -> pd.DataFrame:
    """Compute per-turn plan drift (cosine similarity to initial plan) for ALL model traces.

    Works for both dags and nodags models — uses pure reasoning text (DAGs and code stripped).
    Returns long-format DataFrame: model, base_model, mode, comp, turn_idx, n_turns,
    norm_turn, sim_to_start, drift, success.
    """
    import pickle
    cache = _load_embed_cache()
    embedder = _get_embedder()

    # ── Pre-fetch all turn embeddings async ───────────────────────────────────
    if embedder is not None:
        drift_recs = load_drift_records()
        pending: list[tuple] = []
        for rec in drift_recs:
            m, c = rec["model"], rec["comp"]
            for i, turn in enumerate(rec["trajectory"]):
                if turn.get("role") != "assistant": continue
                text = _strip_dag_and_code(turn.get("content", "") or "")
                if text: pending.append((f"drift|{m}|{c}|{i}", [text]))
        _prefetch_embeddings(pending, embedder, cache, label="📈 Drift embeddings")
    else:
        drift_recs = load_drift_records()

    new_entries = 0

    def _emb_single(key: str, text: str):
        nonlocal new_entries
        has_content = bool(text and text.strip())
        if key in cache and (cache[key] is not None or not has_content):
            return cache[key]
        if not has_content or embedder is None:
            cache[key] = None
            return None
        vec = _mean_embed(embedder, [text])
        cache[key] = vec
        new_entries += 1
        if new_entries % 20 == 0:
            _save_embed_cache(cache)
        return vec

    rows = []
    for rec in drift_recs:
        traj   = rec["trajectory"]
        model  = rec["model"]
        comp   = rec["comp"]
        mode   = rec["mode"]

        # Collect one text per assistant turn (DAGs + code stripped)
        turn_texts: list[str] = []
        for turn in traj:
            if turn.get("role") != "assistant": continue
            content = turn.get("content", "") or ""
            text = _strip_dag_and_code(content)
            if text:
                turn_texts.append(text)

        if len(turn_texts) < 2: continue   # need anchor + at least one more

        n = len(turn_texts)
        turn_embs = [
            _emb_single(f"drift|{model}|{comp}|{i}", t)
            for i, t in enumerate(turn_texts)
        ]
        anchor = turn_embs[0]
        if anchor is None: continue

        for i, emb in enumerate(turn_embs):
            sim = _cosine(anchor, emb) if emb is not None else None
            rows.append({
                "model":        model,
                "base_model":   rec["base_model"],
                "mode":         mode,
                "comp":         comp,
                "turn_idx":     i,
                "n_turns":      n,
                "norm_turn":    i / max(1, n - 1),
                "sim_to_start": sim,
                "drift":        (1.0 - sim) if sim is not None else None,
                "success":      rec["success"],
            })

    if new_entries:
        _save_embed_cache(cache)

    return pd.DataFrame(rows)


def page_dag_consistency(records):
    st.header("🔬 DAG Consistency Analysis")
    apply_style()

    if not records:
        st.warning("No per-task *_dags.json files found in evaluation_results/."); return

    df_all = pd.DataFrame([
        {k:v for k,v in r.items()
         if k not in ("r_kw","i_kw","r_lbls","i_lbls","all_code","trajectory","all_r_dags","all_i_dags")}
        for r in records
    ])

    all_models = sorted(df_all["model"].unique())
    sel_models = st.multiselect("Models", all_models, default=all_models, key="cons_models")
    if not sel_models: return
    df_all = df_all[df_all["model"].isin(sel_models)].copy()

    # ── Protocol Compliance Banner ─────────────────────────────────────────────
    st.markdown("### 📋 DAG Protocol Compliance")
    st.caption(
        "Models running in **dags** mode are expected to emit a reasoning DAG every turn. "
        "The **penalised success rate** only counts a run as successful if the model also produced a DAG. "
        "Models that complete the task without emitting a DAG gain an unfair advantage by skipping the planning overhead."
    )

    compliance_rows = []
    for m in sel_models:
        sub = df_all[df_all["model"] == m]
        total = len(sub)
        if total == 0: continue
        n_dag        = sub["has_dag"].sum()
        success_col  = sub["outcome"].eq("complete")
        n_success    = success_col.sum()
        n_dag_succ   = (sub["has_dag"] & success_col).sum()
        emit_rate    = n_dag / total
        raw_succ     = n_success / total
        pen_succ     = n_dag_succ / total          # penalised: must have DAG + be successful
        penalty      = raw_succ - pen_succ         # how much the model "cheated"
        compliance_rows.append({
            "Model":                  m,
            "Runs":                   total,
            "DAG emission rate":      emit_rate,
            "Raw success rate":       raw_succ,
            "Penalised success rate": pen_succ,
            "Penalty (raw − pen)":    penalty,
        })

    df_comp = pd.DataFrame(compliance_rows).set_index("Model")

    # Style: highlight bad compliance in red
    def _style_compliance(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for idx in df.index:
            er = df.loc[idx, "DAG emission rate"]
            pen = df.loc[idx, "Penalty (raw − pen)"]
            if er < 0.5:
                styles.loc[idx, "DAG emission rate"] = "background-color:#FFCDD2; color:#B71C1C; font-weight:bold"
            elif er < 0.8:
                styles.loc[idx, "DAG emission rate"] = "background-color:#FFE0B2; color:#E65100"
            else:
                styles.loc[idx, "DAG emission rate"] = "background-color:#C8E6C9; color:#1B5E20"
            if pen > 0.1:
                styles.loc[idx, "Penalty (raw − pen)"] = "background-color:#FFCDD2; color:#B71C1C; font-weight:bold"
        return styles

    st.dataframe(
        df_comp.style
               .format({
                   "DAG emission rate":      "{:.1%}",
                   "Raw success rate":       "{:.1%}",
                   "Penalised success rate": "{:.1%}",
                   "Penalty (raw − pen)":    "{:+.1%}",
               })
               .apply(_style_compliance, axis=None),
        use_container_width=True,
    )

    # DAG emission rate only
    fig, ax = plt.subplots(figsize=(max(14, len(sel_models) * 1.8), 9))
    fig.suptitle("DAG Protocol Compliance", fontsize=14, fontweight="bold")
    emit_vals = [df_comp.loc[m, "DAG emission rate"] if m in df_comp.index else 0 for m in sel_models]
    bar_colors = ["#F44336" if v < 0.5 else "#FF9800" if v < 0.8 else "#4CAF50" for v in emit_vals]
    bars = ax.bar(sel_models, emit_vals, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(sel_models))); ax.set_xticklabels(sel_models, rotation=45, ha="right")
    ax.axhline(1.0, color="green", ls="--", lw=1.5, alpha=0.5, label="100% (ideal)")
    ax.axhline(0.5, color="red",   ls="--", lw=1.5, alpha=0.5, label="50% (warning)")
    ax.set_ylabel("DAG emission rate"); ax.set_title("DAG Emission Rate by Model\n(🔴 < 50%  🟠 50–80%  🟢 ≥ 80%)")
    ax.set_ylim(0, 1.15); ax.legend()
    for bar, v in zip(bars, emit_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout(); fig_to_st(fig)
    st.divider()

    df_dag = df_all[df_all["has_ri_dag"]].copy()   # R→I tabs: need both r and i dags
    df_dag = df_dag[df_dag["model"].isin(sel_models)]
    colors = [DAG_MODEL_CONFIGS.get(m, {}).get("color", "#888888") for m in sel_models]
    positions = np.arange(len(sel_models))

    tab_ri,tab_rouge,tab_cov,tab_cross,tab_vocab,tab_inspect,tab_sem,tab_rev,tab_drift,tab_fail = st.tabs(
        ["R→I Consistency","ROUGE","Code Coverage","Cross-Model","Vocab Shift","Inspect Run",
         "🧠 Semantic Embeddings","🔁 Revision Patterns","📈 Plan Drift","🔴 Failure Analysis"])

    # ── R→I ──
    with tab_ri:
        metrics=[("ri_jaccard","R→I Jaccard",False),("ri_overlap","R→I Overlap",False),
                 ("node_delta","Node Δ (impl−reason)",True),("code_cov","Code Coverage",False)]
        fig,axes=plt.subplots(1,4,figsize=(26, 8))
        fig.suptitle("Intra-model R→I Consistency",fontsize=13,fontweight="bold")
        for ax,(col,ylabel,show_zero) in zip(axes,metrics):
            if col not in df_dag.columns: ax.set_title(col+"\n(no data)"); ax.axis("off"); continue
            vals=[df_dag[df_dag["model"]==m][col].dropna().values for m in sel_models]
            bp=ax.boxplot(vals,positions=positions,patch_artist=True,widths=0.5,
                          flierprops=dict(marker=".",alpha=0.4,markersize=4))
            for patch,color in zip(bp["boxes"],colors): patch.set_facecolor(color); patch.set_alpha(0.65)
            for pos,v in zip(positions,vals):
                mn=np.mean(v) if len(v) else 0
                ax.plot(pos,mn,"D",color="black",markersize=6,zorder=5)
                ax.text(pos,mn+0.02,f"{mn:.3f}",ha="center",va="bottom",fontsize=8)
            if show_zero: ax.axhline(0,color="red",lw=1,ls="--",alpha=0.5)
            ax.set_xticks(positions); ax.set_xticklabels(sel_models,rotation=90, ha="right",fontsize=8)
            ax.set_title(ylabel)
        plt.tight_layout(); fig_to_st(fig)

        rows=[]
        for m in sel_models:
            for outcome in ["complete","error","no_submit","ALL"]:
                s=df_dag[df_dag["model"]==m] if outcome=="ALL" else df_dag[(df_dag["model"]==m)&(df_dag["outcome"]==outcome)]
                if len(s)==0: continue
                rows.append({"Model":m,"Outcome":outcome,"n":len(s),
                    "R nodes":    s["r_nodes"].mean()   if "r_nodes"    in s else None,
                    "I nodes":    s["i_nodes"].mean()   if "i_nodes"    in s else None,
                    "R→I Jaccard":s["ri_jaccard"].mean()if "ri_jaccard" in s else None,
                    "R→I Overlap":s["ri_overlap"].mean()if "ri_overlap" in s else None,
                    "Code Coverage":s["code_cov"].mean()if "code_cov"   in s else None})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index(["Model","Outcome"]).round(3),use_container_width=True)

    # ── ROUGE ──
    with tab_rouge:
        rouge_metrics=[("ri_rouge1","ROUGE-1\n(unigram)"),("ri_rouge2","ROUGE-2\n(bigram)"),("ri_rougeL","ROUGE-L\n(sequence)")]
        fig,axes=plt.subplots(1,3,figsize=(24, 8))
        for ax,(metric,label) in zip(axes,rouge_metrics):
            if metric not in df_dag.columns: ax.set_title(metric+"\n(no data)"); ax.axis("off"); continue
            vals=[df_dag[df_dag["model"]==m][metric].dropna().values for m in sel_models]
            bp=ax.boxplot(vals,positions=positions,patch_artist=True,widths=0.5,
                          flierprops=dict(marker=".",alpha=0.4,markersize=4))
            for patch,color in zip(bp["boxes"],colors): patch.set_facecolor(color); patch.set_alpha(0.65)
            for pos,v in zip(positions,vals):
                mn=np.nanmean(v) if len(v) else 0
                ax.plot(pos,mn,"D",color="black",markersize=6,zorder=5)
                ax.text(pos,mn+0.015,f"{mn:.3f}",ha="center",va="bottom",fontsize=8)
            ax.set_xticks(positions); ax.set_xticklabels(sel_models,rotation=90, ha="right",fontsize=8)
            ax.set_ylabel("F1"); ax.set_title(label.split("\n")[0]); ax.set_ylim(0,1.1)
        plt.tight_layout(); fig_to_st(fig)

        sel_sc=st.selectbox("Scatter model",sel_models,key="rouge_sc")
        fig2,ax2=plt.subplots(figsize=(8, 6))
        sub=df_dag[df_dag["model"]==sel_sc].dropna(subset=["ri_rouge1","ri_rougeL"])
        for outcome,marker in [("complete","o"),("error","^"),("no_submit","x")]:
            s=sub[sub["outcome"]==outcome]
            ax2.scatter(s["ri_rouge1"],s["ri_rougeL"],marker=marker,alpha=0.7,s=40,label=outcome)
        ax2.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4,label="y=x")
        ax2.set_xlabel("ROUGE-1"); ax2.set_ylabel("ROUGE-L")
        ax2.set_title(f"{sel_sc}: ROUGE-1 vs ROUGE-L"); ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.legend(fontsize=8)
        fig_to_st(fig2)

    # ── Code Coverage ──
    with tab_cov:
        if "code_cov" not in df_dag.columns: st.info("No code coverage data."); return
        fig,axes=plt.subplots(1,2,figsize=(18, 8))
        ax=axes[0]
        for m,color in zip(sel_models,colors):
            v=df_dag[df_dag["model"]==m]["code_cov"].dropna().values
            if len(v): ax.hist(v,bins=20,range=(0,1),alpha=0.55,label=f"{m} (μ={v.mean():.3f})",color=color)
        ax.set_xlabel("Coverage rate"); ax.set_ylabel("# runs"); ax.set_title("Node coverage distribution"); ax.legend(fontsize=8)

        ax=axes[1]; x2=np.arange(len(sel_models)); full_pcts=[]; low_pcts=[]
        for m in sel_models:
            v=df_dag[df_dag["model"]==m]["code_cov"].dropna().values
            full_pcts.append(100*sum(vv==1.0 for vv in v)/max(1,len(v)))
            low_pcts.append( 100*sum(vv<0.8  for vv in v)/max(1,len(v)))
        b1=ax.bar(x2-0.175,full_pcts,0.35,label="100% coverage",color="#43A047",alpha=0.8)
        b2=ax.bar(x2+0.175,low_pcts, 0.35,label="<80% coverage", color="#E53935",alpha=0.8)
        for bar,v in zip([*b1,*b2],full_pcts+low_pcts):
            if v>1: ax.text(bar.get_x()+bar.get_width()/2,v+0.5,f"{v:.0f}%",ha="center",va="bottom",fontsize=9)
        ax.set_xticks(x2); ax.set_xticklabels(sel_models,rotation=90, ha="right")
        ax.set_ylabel("% of runs"); ax.set_title("Full vs partial coverage"); ax.legend()
        plt.tight_layout(); fig_to_st(fig)

        if "first_miss" in df_dag.columns:
            st.subheader("Position of First Uncovered Node")
            fig3,ax3=plt.subplots(figsize=(12, 7))
            for m,color in zip(sel_models,colors):
                v=df_dag[df_dag["model"]==m]["first_miss"].dropna().values
                if len(v): ax3.hist(v,bins=15,range=(0,1),alpha=0.55,
                                    label=f"{m} (n={len(v)}, μ={v.mean():.2f})",color=color)
            ax3.set_xlabel("Fractional position in plan"); ax3.set_ylabel("# runs"); ax3.legend(fontsize=8)
            fig_to_st(fig3)

    # ── Cross-Model ──
    with tab_cross:
        st.subheader("Inter-model Plan Similarity (Jaccard)")
        by_r: dict = defaultdict(dict); by_i: dict = defaultdict(dict)
        for r in records:
            if r["has_dag"] and r["model"] in sel_models:
                by_r[r["comp"]][r["model"]]=r["r_kw"]
                by_i[r["comp"]][r["model"]]=r["i_kw"]
        pairs_list=[(a,b) for i,a in enumerate(sel_models) for b in sel_models[i+1:]]
        cross_rows=[]
        for comp in by_r:
            if len(by_r[comp])<2: continue
            for m1,m2 in pairs_list:
                if m1 not in by_r[comp] or m2 not in by_r[comp]: continue
                ar,br=by_r[comp][m1],by_r[comp][m2]
                ai,bi=by_i[comp][m1],by_i[comp][m2]
                cross_rows.append({"comp":comp,"m1":m1,"m2":m2,
                    "r_jaccard":_jaccard(ar,br),"i_jaccard":_jaccard(ai,bi)})
        if not cross_rows:
            st.info("Need ≥2 models with shared competitions.")
        else:
            cross_df=pd.DataFrame(cross_rows)
            fig,axes=plt.subplots(1,2,figsize=(18, 8))
            fig.suptitle("Inter-model Plan Similarity",fontsize=12,fontweight="bold")
            for ax,(plan,title) in zip(axes,[("r_jaccard","Reasoning DAG"),("i_jaccard","Implementation DAG")]):
                mat=pd.DataFrame(index=sel_models,columns=sel_models,dtype=float)
                for m in sel_models: mat.loc[m,m]=1.0
                for _,row in cross_df.iterrows():
                    mat.loc[row["m1"],row["m2"]]=row[plan]; mat.loc[row["m2"],row["m1"]]=row[plan]
                sns.heatmap(mat.astype(float),ax=ax,annot=True,fmt=".3f",cmap="YlOrRd",
                            vmin=0,vmax=0.5,linewidths=0.5,cbar_kws={"label":"Jaccard"})
                ax.set_title(title); ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha="right")
            plt.tight_layout(); fig_to_st(fig)
            pr=cross_df.groupby(["m1","m2"])["r_jaccard"].mean().reset_index()
            pi=cross_df.groupby(["m1","m2"])["i_jaccard"].mean().reset_index()
            mg=pr.merge(pi,on=["m1","m2"]); mg.columns=["Model A","Model B","Avg R Jaccard","Avg I Jaccard"]
            mg["n competitions"]=cross_df.groupby(["m1","m2"]).size().values
            st.dataframe(mg.round(3),use_container_width=True)

    # ── Vocab Shift ──
    with tab_vocab:
        sel_v=st.selectbox("Model",sel_models,key="vocab_m")
        recs_m=[r for r in records if r["model"]==sel_v and r["has_dag"]]
        if not recs_m: st.info("No DAG records for this model.")
        else:
            added:Counter=Counter(); dropped:Counter=Counter()
            for r in recs_m:
                for w in r["i_kw"]-r["r_kw"]: added[w]+=1
                for w in r["r_kw"]-r["i_kw"]: dropped[w]+=1
            top_n=st.slider("Top N words",5,30,15,key="vocab_n")
            fig,axes=plt.subplots(1,2,figsize=(18, 8))
            color=DAG_MODEL_CONFIGS.get(sel_v,{}).get("color","#2196F3")
            for ax,(ctr,label,clr) in zip(axes,[
                (added,  "Added in implementation",  color),
                (dropped,"Dropped from reasoning",   "#888888")
            ]):
                top=ctr.most_common(top_n); words=[w for w,_ in top][::-1]; cnts=[c for _,c in top][::-1]
                ax.barh(words,cnts,color=clr,alpha=0.75)
                ax.set_title(f"{sel_v} — {label}",fontsize=10); ax.set_xlabel("# competitions")
            plt.tight_layout(); fig_to_st(fig)

    # ── Inspect Run ──
    with tab_inspect:
        sel_im=st.selectbox("Model",sel_models,key="insp_m")
        recs_m=[r for r in records if r["model"]==sel_im and r["has_dag"]]
        if not recs_m: st.info("No records.")
        else:
            comps=[r["comp"] for r in recs_m]
            sel_ic=st.selectbox("Competition",comps,key="insp_comp")
            rec=next((r for r in recs_m if r["comp"]==sel_ic),None)
            if rec:
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Outcome",rec["outcome"])
                c2.metric("R→I Jaccard",f"{rec.get('ri_jaccard',0):.3f}")
                c3.metric("Code Coverage",f"{rec.get('code_cov',0):.3f}")
                c4.metric("Node Δ",str(rec.get("node_delta","—")))

                st.subheader("Reasoning nodes")
                for i,lbl in enumerate(rec.get("r_lbls",[]),1): st.write(f"{i}. {lbl}")

                st.subheader("Implementation nodes")
                all_code=rec.get("all_code",""); code_set=set(re.findall(r"[a-z][a-z0-9]+",all_code))
                for i,lbl in enumerate(rec.get("i_lbls",[]),1):
                    nkw={w for w in re.findall(r"[a-z][a-z0-9]+",lbl) if w not in STOP_WORDS and len(w)>2}
                    in_code=bool(nkw & code_set); new_node=not bool(nkw & rec.get("r_kw",set()))
                    tag="✅" if in_code else "❌"; nm=" 🆕" if new_node else ""
                    st.write(f"{i}. {tag} {lbl}{nm}")

                if rec["all_r_dags"] and rec["all_i_dags"]:
                    fig,axes=plt.subplots(1,2,figsize=(18, 8))
                    draw_dag_on_ax(rec["all_r_dags"][0],axes[0],f"Reasoning ({rec.get('r_nodes','?')} nodes)")
                    draw_dag_on_ax(rec["all_i_dags"][0],axes[1],f"Implementation ({rec.get('i_nodes','?')} nodes)")
                    fig.suptitle(f"{sel_im} · {sel_ic}",fontsize=11,fontweight="bold")
                    plt.tight_layout(); fig_to_st(fig)

                if rec.get("n_reasoning_dags",0)>1 or rec.get("n_implementation_dags",0)>1:
                    st.subheader(f"All DAGs for this run ({rec['n_reasoning_dags']} reasoning, {rec['n_implementation_dags']} implementation)")
                    all_d=[(d,"Reasoning") for d in rec["all_r_dags"]]+[(d,"Impl") for d in rec["all_i_dags"]]
                    nc=min(3,len(all_d)); nr=(len(all_d)+nc-1)//nc
                    fig,axes2=plt.subplots(nr,nc,figsize=(7*nc,5*nr))
                    flat=np.array(axes2).flatten() if nr*nc>1 else [axes2]
                    for i,(d,dl) in enumerate(all_d): draw_dag_on_ax(d,flat[i],f"{dl} #{i+1}")
                    for i in range(len(all_d),len(flat)): flat[i].axis("off")
                    plt.tight_layout(); fig_to_st(fig)

    # ── Semantic Embeddings ───────────────────────────────────────────────────
    with tab_sem:
        st.subheader("🧠 Semantic Embedding Analysis")

        fp = _records_fingerprint(records)
        embedder_available = _get_embedder() is not None
        if not embedder_available:
            st.warning("⚠️ `sentence-transformers` not available — install it to enable semantic analysis.\n\n"
                       "```bash\npip install sentence-transformers\n```")

        with st.spinner("Computing embeddings (cached after first run)…"):
            df_sem, df_rev, _ = compute_semantic_df(fp)

        if df_sem.empty:
            st.info("No semantic data computed yet."); 
        else:
            df_sem_f = df_sem[df_sem["model"].isin(sel_models)]
            positions_s = np.arange(len(sel_models))

            # ── Box plots: three semantic metrics ────────────────────────────
            st.markdown("#### Cosine similarity between plan representations")
            sem_metrics = [
                ("r_i_semantic",    "R→I Semantic\n(reasoning→implementation)"),
                ("r_code_semantic", "Reasoning→Code\nSemantic"),
                ("i_code_semantic", "Implementation→Code\nSemantic"),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(32, 16))
            fig.suptitle("Semantic Similarity Analysis (Embedding-Based)", fontsize=15, fontweight="bold")
            for ax, (col, label) in zip(axes, sem_metrics):
                vals = [df_sem_f[df_sem_f["model"]==m][col].dropna().values for m in sel_models]
                bp = ax.boxplot(vals, positions=positions_s, patch_artist=True, widths=0.5,
                                flierprops=dict(marker=".", alpha=0.4, markersize=4))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color); patch.set_alpha(0.65)
                for pos, v in zip(positions_s, vals):
                    mn = np.nanmean(v) if len(v) else 0
                    ax.plot(pos, mn, "D", color="black", markersize=6, zorder=5)
                    ax.text(pos, mn+0.015, f"{mn:.3f}", ha="center", va="bottom", fontsize=14)
                ax.set_xticks(positions_s); ax.set_xticklabels(sel_models, rotation=90, ha="right", fontsize=14)
                ax.set_ylabel("Cosine similarity"); ax.set_title(label.split("\n")[0])
                ax.set_ylim(0, 1.05)
                ax.axhline(0.5, color="red", ls="--", alpha=0.4, lw=1)
            plt.tight_layout(); fig_to_st(fig)

            # ── DAG→Code semantic by success ─────────────────────────────────
            # Uses i_code for models with impl dag, r_code for adaptive (reasoning-only)
            st.markdown("#### DAG→Code similarity by outcome")
            st.caption("`Impl→Code` for standard models · `Reasoning→Code` for adaptive models (no impl dag)")
            n_models = len(sel_models)
            n_cols = min(3, n_models)
            n_rows = math.ceil(n_models / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(10 * n_cols, 10 * n_rows),
                                     sharey=True, squeeze=False)
            axes_flat = axes.flatten()
            for ax, m, color in zip(axes_flat, sel_models, colors):
                sub = df_sem_f[df_sem_f["model"] == m].dropna(subset=["code_semantic", "success"])
                succ = sub[sub["success"]]["code_semantic"]
                fail = sub[~sub["success"]]["code_semantic"]
                ax.scatter(["Success"] * len(succ), succ, c="green", alpha=0.6, s=80, label="Success")
                ax.scatter(["Failure"] * len(fail), fail, c="red",   alpha=0.6, s=80, marker="x", label="Failure")
                if len(succ): ax.hlines(succ.mean(), -0.35, 0.35, colors="green", lw=3)
                if len(fail): ax.hlines(fail.mean(), 0.65, 1.35, colors="red",   lw=3)
                has_impl = sub["has_impl_dag"].any() if "has_impl_dag" in sub.columns else False
                metric_lbl = "Impl→Code" if has_impl else "R→Code"
                ax.set_title(f"{m}\n({metric_lbl})", fontsize=13, fontweight="bold")
                ax.set_ylim(0, 1); ax.tick_params(labelsize=12)
                ax.set_ylabel("Cosine similarity to code", fontsize=12)
            # Hide unused axes
            for ax in axes_flat[n_models:]:
                ax.set_visible(False)
            fig.suptitle("DAG→Code semantic similarity by outcome", fontsize=16, fontweight="bold")
            plt.tight_layout(); fig_to_st(fig)

            st.markdown("#### Summary table")
            rows = []
            for m in sel_models:
                sub = df_sem_f[df_sem_f["model"] == m]
                s   = sub[sub["success"]]; f = sub[~sub["success"]]
                has_impl = sub["has_impl_dag"].any() if "has_impl_dag" in sub.columns else False
                rows.append({
                    "Model":             m,
                    "DAG type":          "R+I" if has_impl else "R only",
                    "R→I Sem":           round(sub["r_i_semantic"].mean(),   3) if sub["r_i_semantic"].notna().any()    else None,
                    "R→Code Sem":        round(sub["r_code_semantic"].mean(), 3) if sub["r_code_semantic"].notna().any() else None,
                    "I→Code Sem":        round(sub["i_code_semantic"].mean(), 3) if sub["i_code_semantic"].notna().any() else None,
                    "DAG→Code (success)":round(s["code_semantic"].mean(),     3) if s["code_semantic"].notna().any()    else None,
                    "DAG→Code (fail)":   round(f["code_semantic"].mean(),     3) if f["code_semantic"].notna().any()    else None,
                    "Δ (succ−fail)":     round(s["code_semantic"].mean() - f["code_semantic"].mean(), 3)
                                         if s["code_semantic"].notna().any() and f["code_semantic"].notna().any() else None,
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

            # ── Code grounding verdict ────────────────────────────────────────
            st.markdown("#### Code grounding: Is DAG plan reflected in code?")
            for m in sel_models:
                sub = df_sem_f[df_sem_f["model"] == m]
                r_c = sub["r_code_semantic"].mean()
                i_c = sub["i_code_semantic"].mean() if sub["i_code_semantic"].notna().any() else None
                if pd.isna(r_c):
                    st.write(f"**{m}**: no data"); continue
                if i_c is not None and not pd.isna(i_c):
                    delta = (i_c - r_c) / r_c * 100 if r_c > 0 else 0
                    icon = "✅" if i_c > r_c else "❌"
                    st.write(f"{icon} **{m}** — R→Code: `{r_c:.3f}` · I→Code: `{i_c:.3f}` · Δ: `{delta:+.1f}%`")
                else:
                    st.write(f"🔵 **{m}** (adaptive, R only) — R→Code: `{r_c:.3f}`")

    # ── Revision Patterns ─────────────────────────────────────────────────────
    with tab_rev:
        st.subheader("🔁 Trajectory Revision Patterns")
        st.caption("Based on raw reasoning text and code extracted from trajectories — works even without DAGs.")

        fp = _records_fingerprint(records)
        with st.spinner("Computing trajectory embeddings (cached after first run)…"):
            _, df_rev, df_im = compute_semantic_df(fp)

        if df_rev.empty:
            st.info("No trajectory data found."); 
        else:
            df_rf = df_rev[df_rev["model"].isin(sel_models)].copy()

            # ── 1. Plan stability: consecutive-turn cosine similarity ─────────
            st.markdown("#### Plan stability — consecutive turn cosine similarity")
            st.caption("High similarity = model refines incrementally. Low similarity = bigger plan pivots.")
            fig, axes = plt.subplots(1, 2, figsize=(24, 12))
            fig.suptitle("Trajectory Revision Patterns (text/code embedding)", fontsize=13, fontweight="bold")

            ax = axes[0]
            vals_by_m = [df_rf[df_rf["model"]==m]["mean_consec_sim"].dropna().values for m in sel_models]
            bp = ax.boxplot(vals_by_m, patch_artist=True, widths=0.5,
                            flierprops=dict(marker=".", alpha=0.4, markersize=4))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.65)
            for pos, vals in enumerate(vals_by_m):
                if len(vals): ax.text(pos+1, np.nanmean(vals)+0.01, f"{np.nanmean(vals):.3f}", ha="center", fontsize=8)
            ax.set_xticks(range(1, len(sel_models)+1))
            ax.set_xticklabels(sel_models, rotation=90, ha="right", fontsize=8)
            ax.set_ylabel("Mean consecutive cosine sim"); ax.set_title("Plan Stability\n(higher = more incremental)")
            ax.set_ylim(0, 1.05); ax.axhline(0.5, color="red", ls="--", alpha=0.3, lw=1)

            # ── 2. First → last plan drift ────────────────────────────────────
            ax = axes[1]
            vals_by_m3 = [df_rf[df_rf["model"]==m]["first_last_drift"].dropna().values for m in sel_models]
            bp3 = ax.boxplot(vals_by_m3, patch_artist=True, widths=0.5,
                             flierprops=dict(marker=".", alpha=0.4, markersize=4))
            for patch, color in zip(bp3["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.65)
            for pos, vals in enumerate(vals_by_m3):
                if len(vals): ax.text(pos+1, np.nanmean(vals)+0.01, f"{np.nanmean(vals):.3f}", ha="center", fontsize=8)
            ax.set_xticks(range(1, len(sel_models)+1))
            ax.set_xticklabels(sel_models, rotation=90, ha="right", fontsize=8)
            ax.set_ylabel("Drift (1 − cosine)"); ax.set_title("First → Last Plan Drift\n(exploration breadth)")
            ax.set_ylim(0, 1.05); ax.axhline(0.5, color="red", ls="--", alpha=0.3, lw=1)
            plt.tight_layout(); fig_to_st(fig)

            # ── 3. Drift vs success ───────────────────────────────────────────
            st.markdown("#### Plan drift vs outcome")
            fig, ax = plt.subplots(figsize=(13, 11))
            for m, color in zip(sel_models, colors):
                sub = df_rf[df_rf["model"]==m].dropna(subset=["mean_consec_drift","success"])
                s = sub[sub["success"]]["mean_consec_drift"]
                f = sub[~sub["success"]]["mean_consec_drift"]
                ax.scatter(["✅ Success"]*len(s) if len(s) else [], s, c=color, alpha=0.6, s=40, label=m if len(s) else None)
                ax.scatter(["❌ Failure"]*len(f) if len(f) else [], f, c=color, alpha=0.6, s=40, marker="x")
            ax.set_ylabel("Mean consecutive drift"); ax.set_title("Plan Consistency by Outcome")
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout(); fig_to_st(fig)

            # ── 5. Inter-model similarity heatmap ─────────────────────────────
            st.markdown("#### Inter-model similarity (same competition, final plan)")
            st.caption("How similar are different models' final reasoning for the same challenge? "
                       "Low = models explore very different strategies.")
            if not df_im.empty:
                df_im_f = df_im[df_im["model_a"].isin(sel_models) & df_im["model_b"].isin(sel_models)]
                if not df_im_f.empty:
                    # Build symmetric matrix
                    avg_sim = df_im_f.groupby(["model_a","model_b"])["cos_sim"].mean().reset_index()
                    all_m   = sorted(set(avg_sim["model_a"]) | set(avg_sim["model_b"]))
                    mat     = pd.DataFrame(np.nan, index=all_m, columns=all_m, dtype=float)
                    for m_idx in range(len(all_m)):
                        mat.iloc[m_idx, m_idx] = 1.0
                    for _, row in avg_sim.iterrows():
                        mat.loc[row["model_a"], row["model_b"]] = row["cos_sim"]
                        mat.loc[row["model_b"], row["model_a"]] = row["cos_sim"]

                    fig, ax = plt.subplots(figsize=(max(24, len(all_m)*3.0), max(18, len(all_m)*2)))
                    sns.heatmap(mat.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
                                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                                xticklabels=all_m, yticklabels=all_m)
                    ax.set_title("Avg inter-model cosine similarity\n(same competition, final reasoning)")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=8)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
                    plt.tight_layout(); fig_to_st(fig)

                    # Per-pair box plot across competitions
                    st.markdown("#### Similarity distribution across competitions (per pair)")
                    fig, ax = plt.subplots(figsize=(max(18, len(df_im_f["pair"].unique())*1.6), 12))
                    pairs = sorted(df_im_f["pair"].unique())
                    pair_data = [df_im_f[df_im_f["pair"]==p]["cos_sim"].dropna().values for p in pairs]
                    bp = ax.boxplot(pair_data, patch_artist=True, widths=0.6,
                                   flierprops=dict(marker=".", alpha=0.4))
                    for patch in bp["boxes"]: patch.set_facecolor("#64B5F6"); patch.set_alpha(0.7)
                    ax.set_xticks(range(1, len(pairs)+1))
                    ax.set_xticklabels(pairs, rotation=90, ha="right", fontsize=7)
                    ax.set_ylabel("Cosine similarity"); ax.set_title("Inter-model similarity by pair (across competitions)")
                    ax.axhline(0.5, color="red", ls="--", alpha=0.4, lw=1)
                    plt.tight_layout(); fig_to_st(fig)
                else:
                    st.info("Not enough model overlap to compute inter-model similarities for selected models.")

            # ── Summary table ─────────────────────────────────────────────────
            st.markdown("#### Per-model revision summary")
            rows = []
            for m in sel_models:
                sub = df_rf[df_rf["model"]==m]
                rows.append({
                    "Model": m,
                    "Runs": len(sub),
                    "Success %": f"{100*sub['success'].mean():.1f}%" if len(sub) else "—",
                    "Avg turns": round(sub["n_turns"].mean(), 1) if len(sub) else None,
                    "Runs w/ revisions %": f"{100*(sub['n_revisions']>0).mean():.0f}%" if len(sub) else "—",
                    "Avg consec sim": round(sub["mean_consec_sim"].mean(), 3) if sub["mean_consec_sim"].notna().any() else None,
                    "First→Last drift": round(sub["first_last_drift"].mean(), 3) if sub["first_last_drift"].notna().any() else None,
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # ── Plan Drift per Turn ──────────────────────────────────────────────────
    with tab_drift:
        st.subheader("📈 Plan Drift per Turn")
        st.caption(
            "Semantic similarity between each assistant turn's **reasoning text** and the **initial plan** (turn 0). "
            "DAG JSON and code are stripped so only natural-language reasoning is compared. "
            "Works for **all models** including nodags. Drift = 1 − similarity."
        )

        if _get_embedder() is None:
            st.warning("sentence-transformers not installed — run `pip install sentence-transformers`.")
        else:
            with st.spinner("Computing per-turn drift embeddings (cached after first run)…"):
                fp_drift = _drift_fingerprint()
                df_drift = compute_drift_df(fp_drift)

            if df_drift.empty:
                st.info("No drift data computed yet.")
            else:
                # All models available in drift df (includes nodags)
                all_drift_models = sorted(df_drift["model"].unique())
                sel_drift = st.multiselect(
                    "Select models (includes nodags variants)",
                    all_drift_models,
                    default=all_drift_models,
                    key="drift_models",
                )
                x_mode = st.radio(
                    "X-axis", ["Normalised turn (0→1)", "Absolute turn index"],
                    horizontal=True, key="drift_xmode",
                )
                use_norm = x_mode == "Normalised turn (0→1)"
                x_col = "norm_turn" if use_norm else "turn_idx"

                df_d = df_drift[df_drift["model"].isin(sel_drift)].copy()
                drift_colors = [_model_color(m) for m in sel_drift]

                # ── 1. Mean similarity curve per model ────────────────────────
                st.markdown("#### Similarity to initial plan over turns")
                st.caption("Mean ± 1 std across all competitions. Turn 0 = initial plan (sim = 1.0 by definition).")

                fig, ax = plt.subplots(figsize=(22, 10))
                if use_norm:
                    bins = np.linspace(0, 1, 21)   # 20 bins
                    bin_labels = (bins[:-1] + bins[1:]) / 2
                    for m, color in zip(sel_drift, drift_colors):
                        sub = df_d[df_d["model"] == m].dropna(subset=["norm_turn", "sim_to_start"])
                        if sub.empty: continue
                        sub = sub.copy()
                        sub["bin"] = pd.cut(sub["norm_turn"], bins=bins, labels=bin_labels)
                        grouped = sub.groupby("bin")["sim_to_start"].agg(["mean", "std"]).reset_index()
                        grouped["bin"] = grouped["bin"].astype(float)
                        ax.plot(grouped["bin"], grouped["mean"], label=m, color=color, lw=2.5, marker="o", ms=4)
                        ax.fill_between(grouped["bin"],
                                        (grouped["mean"] - grouped["std"].fillna(0)).clip(0, 1),
                                        (grouped["mean"] + grouped["std"].fillna(0)).clip(0, 1),
                                        alpha=0.12, color=color)
                    ax.set_xlabel("Normalised turn position (0 = start, 1 = end)", fontsize=11)
                else:
                    max_turn = int(df_d["turn_idx"].max()) if not df_d.empty else 15
                    for m, color in zip(sel_drift, drift_colors):
                        sub = df_d[df_d["model"] == m].dropna(subset=["turn_idx", "sim_to_start"])
                        if sub.empty: continue
                        grouped = sub.groupby("turn_idx")["sim_to_start"].agg(["mean", "std"]).reset_index()
                        ax.plot(grouped["turn_idx"], grouped["mean"], label=m, color=color, lw=2.5, marker="o", ms=4)
                        ax.fill_between(grouped["turn_idx"],
                                        (grouped["mean"] - grouped["std"].fillna(0)).clip(0, 1),
                                        (grouped["mean"] + grouped["std"].fillna(0)).clip(0, 1),
                                        alpha=0.12, color=color)
                    ax.set_xlabel("Turn index", fontsize=11)

                ax.set_ylabel("Cosine similarity to initial plan", fontsize=11)
                ax.set_title("Plan Drift over Trajectory (higher = stays closer to initial plan)", fontsize=13, fontweight="bold")
                ax.set_ylim(0, 1.05)
                ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
                ax.legend(fontsize=8, ncol=2, loc="lower left")
                plt.tight_layout(); fig_to_st(fig)

                # ── 2. Success vs failure drift curves ────────────────────────
                st.markdown("#### Drift curve split by outcome")
                fig, axes = plt.subplots(1, 2, figsize=(30, 11), sharey=True)
                for ax, outcome, label in zip(axes, [True, False], ["✅ Successful runs", "❌ Failed runs"]):
                    sub_out = df_d[df_d["success"] == outcome]
                    for m, color in zip(sel_drift, drift_colors):
                        sub = sub_out[sub_out["model"] == m].dropna(subset=[x_col, "sim_to_start"])
                        if sub.empty: continue
                        if use_norm:
                            bins = np.linspace(0, 1, 21)
                            bin_labels = (bins[:-1] + bins[1:]) / 2
                            sub = sub.copy()
                            sub["bin"] = pd.cut(sub[x_col], bins=bins, labels=bin_labels)
                            g = sub.groupby("bin")["sim_to_start"].mean().reset_index()
                            g["bin"] = g["bin"].astype(float)
                            ax.plot(g["bin"], g["sim_to_start"], label=m, color=color, lw=2, marker="o", ms=3)
                        else:
                            g = sub.groupby(x_col)["sim_to_start"].mean().reset_index()
                            ax.plot(g[x_col], g["sim_to_start"], label=m, color=color, lw=2, marker="o", ms=3)
                    ax.set_title(label, fontsize=12, fontweight="bold")
                    ax.set_xlabel("Normalised turn" if use_norm else "Turn index", fontsize=10)
                    ax.set_ylabel("Sim to initial plan", fontsize=10)
                    ax.set_ylim(0, 1.05)
                    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
                    ax.legend(fontsize=7, ncol=2)
                plt.tight_layout(); fig_to_st(fig)

                # ── 3. Final drift bar chart (first→last similarity) ──────────
                st.markdown("#### Final plan drift by model (first vs last turn similarity)")
                final_rows = []
                for m in sel_drift:
                    sub = df_d[df_d["model"] == m]
                    # For each comp, get the last turn's similarity
                    last_sims = sub.groupby("comp").apply(
                        lambda g: g.sort_values("turn_idx")["sim_to_start"].iloc[-1]
                        if not g.empty else None
                    ).dropna()
                    if last_sims.empty: continue
                    final_rows.append({
                        "model": m,
                        "mean_final_sim": last_sims.mean(),
                        "std_final_sim":  last_sims.std(),
                        "mean_final_drift": 1 - last_sims.mean(),
                        "n": len(last_sims),
                    })
                if final_rows:
                    df_final = pd.DataFrame(final_rows).sort_values("mean_final_drift", ascending=False)
                    fig, ax = plt.subplots(figsize=(max(14, len(df_final)*1.8), 9))
                    bar_colors = [_model_color(r["model"]) for _, r in df_final.iterrows()]
                    bars = ax.bar(range(len(df_final)), df_final["mean_final_drift"],
                                  color=bar_colors, alpha=0.8, edgecolor="white")
                    ax.errorbar(range(len(df_final)), df_final["mean_final_drift"],
                                yerr=df_final["std_final_sim"], fmt="none", color="black", capsize=4, lw=1.5)
                    ax.set_xticks(range(len(df_final)))
                    ax.set_xticklabels(df_final["model"].tolist(), rotation=45, ha="right", fontsize=9)
                    ax.set_ylabel("Mean final drift (1 − sim to initial plan)", fontsize=11)
                    ax.set_title("How far does each model drift from its initial plan?", fontsize=12, fontweight="bold")
                    ax.set_ylim(0, 1.0)
                    for i, (_, row) in enumerate(df_final.iterrows()):
                        ax.text(i, row["mean_final_drift"] + 0.02, f'{row["mean_final_drift"]:.3f}\n(n={row["n"]})',
                                ha="center", va="bottom", fontsize=8)
                    plt.tight_layout(); fig_to_st(fig)

                # ── 4. Summary table ──────────────────────────────────────────
                st.markdown("#### Summary table")
                summary = []
                for m in sel_drift:
                    sub = df_d[df_d["model"] == m].dropna(subset=["sim_to_start"])
                    if sub.empty: continue
                    last = sub.groupby("comp").apply(
                        lambda g: g.sort_values("turn_idx")["sim_to_start"].iloc[-1]
                    ).dropna()
                    summary.append({
                        "Model": m,
                        "Mode": sub["mode"].iloc[0] if not sub.empty else "—",
                        "Runs": sub["comp"].nunique(),
                        "Avg turns": round(sub.groupby("comp")["n_turns"].first().mean(), 1),
                        "Mean final drift": round((1 - last).mean(), 3) if not last.empty else None,
                        "Mean final sim": round(last.mean(), 3) if not last.empty else None,
                        "Success %": f"{100*sub.groupby('comp')['success'].first().mean():.1f}%",
                    })
                st.dataframe(pd.DataFrame(summary).set_index("Model"), use_container_width=True)

    # ── Failure Analysis ──────────────────────────────────────────────────────
    with tab_fail:
        st.subheader("🔴 Failure Analysis")
        st.caption(
            "Structured error categorisation across **all models and modes** "
            "(dags, nodags, adaptive). "
            "Errors are extracted from execution feedback in the trajectory."
        )

        with st.spinner("Loading failure records…"):
            fail_all = load_failure_analysis_records()

        if not fail_all:
            st.info("No per-task JSON files found.")
        else:
            df_fail = pd.DataFrame([{k: v for k, v in r.items() if k != "trajectory"}
                                    for r in fail_all])

            # ── Filters ───────────────────────────────────────────────────────
            all_fail_models = sorted(df_fail["model"].unique())
            sel_fm = st.multiselect("Models", all_fail_models, default=all_fail_models, key="fail_models")
            df_f = df_fail[df_fail["model"].isin(sel_fm)].copy()

            # ── Top-level banner ──────────────────────────────────────────────
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            tc1.metric("Total runs",          len(df_f))
            tc2.metric("✅ Complete",          (df_f["outcome"] == "complete").sum())
            tc3.metric("⚠️ Submission error", (df_f["outcome"] == "error").sum())
            tc4.metric("❌ No submission",    (df_f["outcome"] == "no_submit").sum())
            tc5.metric("Success rate",        f"{100*(df_f['outcome']=='complete').mean():.1f}%")

            # ── Per-model summary table ───────────────────────────────────────
            st.markdown("#### Per-model error breakdown")
            tbl_rows = []
            for m in sel_fm:
                sub = df_f[df_f["model"] == m]
                if sub.empty: continue
                n = len(sub)
                tbl_rows.append({
                    "Model":             m,
                    "Runs":              n,
                    "Success %":         f"{100*(sub['outcome']=='complete').mean():.1f}%",
                    "Fmt errors / run":  round(sub["fmt_errors"].mean(), 2),
                    "Timeouts / run":    round(sub["timeouts"].mean(), 2),
                    "Server errors / run": round(sub["server_errors"].mean(), 2),
                    "Truncations / run": round(sub["truncations"].mean(), 2),
                    "Runtime errors / run": round(sub["runtime_errors"].mean(), 2),
                    "1st-turn code %":   f"{100*sub['first_code'].mean():.0f}%",
                    "1st-turn DAG %":    f"{100*sub['first_dag'].mean():.0f}%",
                    "Recovery rate":     f"{100*sub['recovery_rate'].dropna().mean():.0f}%" if sub['recovery_rate'].notna().any() else "—",
                    "Avg turns":         round(sub["n_turns"].mean(), 1),
                })
            st.dataframe(pd.DataFrame(tbl_rows).set_index("Model"), use_container_width=True)

            # ── Stacked bar: error types per model (per run) ──────────────────
            st.markdown("#### Error frequency per run by model")
            err_cols  = ["fmt_errors", "timeouts", "server_errors",
                         "kernel_errors", "runtime_errors", "truncations"]
            err_labels = ["Format error\n(no code found)", "Timeout",
                          "Server error\n(5xx/disconnect)", "Kernel error",
                          "Runtime error\n(traceback)", "Truncation\n(<python> unclosed)"]
            err_colors = ["#E53935","#FF9800","#9C27B0","#F44336","#607D8B","#00ACC1"]

            model_means = (
                df_f.groupby("model")[err_cols].mean()
                    .loc[[m for m in sel_fm if m in df_f["model"].values]]
            )
            if not model_means.empty:
                fig_e, ax_e = plt.subplots(figsize=(max(16, len(sel_fm)*2), 9))
                bottom = np.zeros(len(model_means))
                for col, label, color in zip(err_cols, err_labels, err_colors):
                    vals = model_means[col].values
                    ax_e.bar(range(len(model_means)), vals, bottom=bottom,
                             label=label, color=color, alpha=0.85, edgecolor="white")
                    bottom += vals
                ax_e.set_xticks(range(len(model_means)))
                ax_e.set_xticklabels(model_means.index.tolist(), rotation=45, ha="right", fontsize=10)
                ax_e.set_ylabel("Average count per run", fontsize=11)
                ax_e.set_title("Execution errors per run (stacked by type)", fontsize=13, fontweight="bold")
                ax_e.legend(loc="upper right", fontsize=9, ncol=2)
                plt.tight_layout(); fig_to_st(fig_e)

            # ── Outcome breakdown per model ────────────────────────────────────
            st.markdown("#### Outcome distribution per model")
            outcome_order = ["complete", "error", "no_submit"]
            outcome_colors = {"complete": "#4CAF50", "error": "#FF9800", "no_submit": "#E53935"}
            pivot = (df_f.groupby(["model", "outcome"]).size()
                         .unstack(fill_value=0)
                         .reindex(columns=outcome_order, fill_value=0))
            pivot = pivot.loc[[m for m in sel_fm if m in pivot.index]]
            if not pivot.empty:
                fig_o, ax_o = plt.subplots(figsize=(max(16, len(sel_fm)*2), 8))
                bottom = np.zeros(len(pivot))
                for outcome in outcome_order:
                    if outcome not in pivot.columns: continue
                    ax_o.bar(range(len(pivot)), pivot[outcome].values, bottom=bottom,
                             label=outcome, color=outcome_colors[outcome], alpha=0.85, edgecolor="white")
                    bottom += pivot[outcome].values
                ax_o.set_xticks(range(len(pivot)))
                ax_o.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right", fontsize=10)
                ax_o.set_ylabel("Number of runs"); ax_o.set_title("Run outcome distribution", fontsize=13, fontweight="bold")
                ax_o.legend(fontsize=10)
                plt.tight_layout(); fig_to_st(fig_o)

            # ── Inspector: browse failed runs ─────────────────────────────────
            st.markdown("#### Failed run inspector")
            st.caption("Inspect individual failed trajectories with error signals highlighted.")

            # ── Outcome pre-filter ────────────────────────────────────────────
            outcome_filter = st.multiselect(
                "Filter by outcome", ["error", "no_submit"],
                default=["error", "no_submit"], key="fail_outcome_filter"
            )
            fail_subset = [r for r in fail_all
                           if r["model"] in sel_fm and r["outcome"] in outcome_filter]
            if not fail_subset:
                st.info("No runs match selected models / outcomes.")
            else:
                insp_models = sorted({r["model"] for r in fail_subset})
                insp_m = st.selectbox("Model", insp_models, key="fail_insp_m")
                recs_fm = [r for r in fail_subset if r["model"] == insp_m]
                recs_fm_sorted = sorted(recs_fm,
                    key=lambda r: (r["outcome"], -(r["fmt_errors"]+r["timeouts"]+r["server_errors"])))
                choices = [
                    f"{r['comp']}  [{r['outcome']}  fmt={r['fmt_errors']} tmout={r['timeouts']} srv={r['server_errors']}]"
                    for r in recs_fm_sorted
                ]
                insp_c = st.selectbox("Competition (sorted by error severity)", choices, key="fail_insp_c")

                if insp_c:
                    comp_name = insp_c.split("  [")[0]
                    frec = next((r for r in recs_fm_sorted if r["comp"] == comp_name), None)
                    if frec:
                        fi1,fi2,fi3,fi4,fi5,fi6 = st.columns(6)
                        fi1.metric("Outcome",     frec["outcome"])
                        fi2.metric("Turns",       frec["n_turns"])
                        fi3.metric("Fmt errors",  frec["fmt_errors"])
                        fi4.metric("Timeouts",    frec["timeouts"])
                        fi5.metric("Srv errors",  frec["server_errors"])
                        fi6.metric("Truncations", frec["truncations"])

                        # ── LLM summarisation ─────────────────────────────────
                        cache_key = f"fail_summary|{frec['model']}|{frec['comp']}"
                        if cache_key not in st.session_state:
                            st.session_state[cache_key] = None

                        if st.button("🤖 Summarise failure with LLM", key="fail_summarise_btn"):
                            traj_sum = frec.get("trajectory", [])
                            # Build a compact transcript of the last 10 turns
                            lines = []
                            for turn in traj_sum[-10:]:
                                if not isinstance(turn, dict): continue
                                role = turn.get("role", "")
                                content = turn.get("content", "") or ""
                                if role == "assistant":
                                    clean = _strip_dag_and_code(content)
                                    lines.append(f"[AGENT]: {clean[:1200]}")
                                elif role == "user":
                                    info = " | ".join(_INFO_RE.findall(content))
                                    lines.append(f"[ENV]: {info[:600]}")
                            transcript = "\n\n".join(lines)
                            prompt = (
                                f"You are analysing a failed AI agent run on a Kaggle competition.\n"
                                f"Competition: {frec['comp']}\n"
                                f"Outcome: {frec['outcome']} (error = Kaggle submission failed, "
                                f"no_submit = agent never produced a submission)\n"
                                f"Error counts: format_errors={frec['fmt_errors']}, "
                                f"timeouts={frec['timeouts']}, server_errors={frec['server_errors']}, "
                                f"truncations={frec['truncations']}\n\n"
                                f"Last ~10 turns of the trajectory (code stripped):\n{transcript}\n\n"
                                f"In 150 words or fewer, diagnose: (1) the primary failure mode, "
                                f"(2) what the agent was attempting when it failed, "
                                f"(3) one concrete fix that could prevent this failure."
                            )
                            with st.spinner("Calling LLM…"):
                                try:
                                    import litellm
                                    resp = litellm.completion(
                                        model="openai/gpt-4.1-mini",
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=300,
                                    )
                                    st.session_state[cache_key] = resp.choices[0].message.content
                                except Exception as exc:
                                    st.session_state[cache_key] = f"Error: {exc}"

                        if st.session_state[cache_key]:
                            st.info(st.session_state[cache_key])

                        # ── Trajectory viewer ─────────────────────────────────
                        traj = frec.get("trajectory", [])
                        n_show = st.slider(
                            "Show last N turns", 1, max(1, len(traj)),
                            min(8, len(traj)), key="fail_n_turns"
                        )
                        for turn in traj[-n_show:]:
                            if not isinstance(turn, dict): continue
                            role    = turn.get("role", "")
                            content = turn.get("content", "") or ""
                            t_idx   = turn.get("turn", "")

                            if role == "assistant":
                                clean = _strip_dag_and_code(content)
                                is_truncated = bool(_PYTHON_OPEN_RE.search(content) and
                                                    not _PYTHON_CLOSE_RE.search(content))
                                label = f"🤖 Assistant turn {t_idx} — {len(content):,} chars"
                                if is_truncated: label += " ✂️ TRUNCATED"
                                with st.expander(label, expanded=True):
                                    if clean.strip():
                                        st.markdown(clean[:2500] + ("…" if len(clean) > 2500 else ""))
                                    if is_truncated:
                                        st.warning("⚠️ Response truncated — `<python>` opened but never closed.")

                            elif role == "user":
                                info_blocks = _INFO_RE.findall(content)
                                is_fmt   = any(_FMT_ERROR_RE.search(b)    for b in info_blocks)
                                is_tmout = any(_TIMEOUT_RE.search(b)       for b in info_blocks)
                                is_srv   = any(_SERVER_ERROR_RE.search(b)  for b in info_blocks)
                                is_rt    = any(_TRACEBACK_RE.search(b)     for b in info_blocks)
                                signals  = (["FORMAT ERR"] if is_fmt   else []) + \
                                           (["TIMEOUT"]    if is_tmout else []) + \
                                           (["SERVER ERR"] if is_srv   else []) + \
                                           (["TRACEBACK"]  if is_rt    else [])
                                label = f"🔧 Execution output turn {t_idx}"
                                if signals: label += "  ⚠️ " + " | ".join(signals)
                                with st.expander(label, expanded=bool(signals)):
                                    for blk in info_blocks:
                                        blk = blk.strip()
                                        if not blk: continue
                                        if any(p.search(blk) for p in
                                               [_FMT_ERROR_RE, _TIMEOUT_RE, _SERVER_ERROR_RE,
                                                _KERNEL_RE, _TRACEBACK_RE]):
                                            st.error(blk[:800])
                                        else:
                                            st.code(blk[:600])

            # ── Side-by-side comparison ───────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Side-by-side run comparison")
            st.caption(
                "Pick both models first — the competition list then shows **only runs where the two models "
                "had different outcomes**, with the outcome of each side shown in the label."
            )

            all_cmp_models = sorted({r["model"] for r in fail_all if r["model"] in sel_fm})
            if len(all_cmp_models) < 1:
                st.info("Need at least one model loaded.")
            else:
                cc1, cc2 = st.columns(2)
                cmp_m_a = cc1.selectbox("Left model",  all_cmp_models, key="cmp_m_a", index=0)
                cmp_m_b = cc2.selectbox("Right model", all_cmp_models, key="cmp_m_b",
                                        index=min(1, len(all_cmp_models)-1))

                # Build outcome lookup: (model, comp) → outcome
                outcome_by = {(r["model"], r["comp"]): r["outcome"] for r in fail_all}

                # Find competitions that both models have run
                comps_a = {r["comp"] for r in fail_all if r["model"] == cmp_m_a}
                comps_b = {r["comp"] for r in fail_all if r["model"] == cmp_m_b}
                shared  = comps_a & comps_b

                outcome_icons = {"complete": "✅", "error": "⚠️", "no_submit": "❌"}

                def _icon(m, c):
                    return outcome_icons.get(outcome_by.get((m, c), ""), "—")

                # Partition into "outcomes differ" vs "same outcome"
                differ = sorted(c for c in shared
                                if outcome_by.get((cmp_m_a, c)) != outcome_by.get((cmp_m_b, c)))
                same   = sorted(c for c in shared
                                if outcome_by.get((cmp_m_a, c)) == outcome_by.get((cmp_m_b, c)))

                show_same = st.checkbox("Also show runs where both had the same outcome", key="cmp_show_same")
                comps_pool = differ + (same if show_same else [])

                if not comps_pool:
                    st.info("No shared competitions found for these two models." if not shared else
                            "Both models had the same outcome on all shared competitions — "
                            "tick 'Also show same outcome' to browse them.")
                else:
                    # Build labelled choices: outcome_a vs outcome_b  competition
                    choices_cmp = [
                        f"{_icon(cmp_m_a, c)} vs {_icon(cmp_m_b, c)}  {c}"
                        for c in comps_pool
                    ]
                    cmp_choice = st.selectbox(
                        f"Competition ({len(differ)} differ, {len(same)} same)",
                        choices_cmp, key="cmp_comp"
                    )
                    cmp_comp = cmp_choice.split("  ", 1)[1] if "  " in cmp_choice else cmp_choice

                    rec_a = next((r for r in fail_all if r["model"] == cmp_m_a and r["comp"] == cmp_comp), None)
                    rec_b = next((r for r in fail_all if r["model"] == cmp_m_b and r["comp"] == cmp_comp), None)

                    h1, h2 = st.columns(2)
                    oa = outcome_by.get((cmp_m_a, cmp_comp), "—")
                    ob = outcome_by.get((cmp_m_b, cmp_comp), "—")
                    h1.markdown(f"**{cmp_m_a}** — {outcome_icons.get(oa,'')} `{oa}`")
                    h2.markdown(f"**{cmp_m_b}** — {outcome_icons.get(ob,'')} `{ob}`")

                    n_cmp = st.slider("Show last N turns (each side)", 1, 20, 6, key="cmp_n_turns")

                    def _render_turns(rec, container):
                        if rec is None:
                            container.info("No run found.")
                            return
                        traj = rec.get("trajectory", [])
                        for turn in traj[-n_cmp:]:
                            if not isinstance(turn, dict): continue
                            role    = turn.get("role", "")
                            content = turn.get("content", "") or ""
                            t_idx   = turn.get("turn", "")
                            if role == "assistant":
                                clean    = _strip_dag_and_code(content)
                                is_trunc = bool(_PYTHON_OPEN_RE.search(content) and
                                                not _PYTHON_CLOSE_RE.search(content))
                                lbl = f"🤖 Turn {t_idx}" + (" ✂️" if is_trunc else "")
                                with container.expander(lbl, expanded=True):
                                    container.markdown(clean[:1500] + ("…" if len(clean) > 1500 else ""))
                                    if is_trunc:
                                        container.warning("Truncated")
                            elif role == "user":
                                info_blocks = _INFO_RE.findall(content)
                                has_err = any(
                                    p.search(b) for b in info_blocks
                                    for p in [_FMT_ERROR_RE, _TIMEOUT_RE,
                                              _SERVER_ERROR_RE, _TRACEBACK_RE]
                                )
                                lbl = f"🔧 Env turn {t_idx}" + ("  ⚠️" if has_err else "")
                                with container.expander(lbl, expanded=has_err):
                                    for blk in info_blocks:
                                        blk = blk.strip()
                                        if not blk: continue
                                        if any(p.search(blk) for p in
                                               [_FMT_ERROR_RE, _TIMEOUT_RE,
                                                _SERVER_ERROR_RE, _TRACEBACK_RE]):
                                            container.error(blk[:600])
                                        else:
                                            container.code(blk[:400])

                    col_a, col_b = st.columns(2)
                    _render_turns(rec_a, col_a)
                    _render_turns(rec_b, col_b)




# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Cross-Benchmark
# ──────────────────────────────────────────────────────────────────────────────
def page_cross_benchmark(dp_data, dp_hard_data, bio_data):
    st.header("⚡ Cross-Benchmark: DAG vs No-DAG Overview")
    apply_style()

    all_models=sorted({
        m for (m,md) in list(dp_data)+list(dp_hard_data)
        if ((m,"dags") in dp_data or (m,"dags") in dp_hard_data)
        and ((m,"nodags") in dp_data or (m,"nodags") in dp_hard_data)
    })


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Cross-Benchmark
# ──────────────────────────────────────────────────────────────────────────────
def page_cross_benchmark(dp_data, dp_hard_data, bio_data):
    st.header("⚡ Cross-Benchmark: DAG vs No-DAG Overview")
    apply_style()

    all_models=sorted({
        m for (m,md) in list(dp_data)+list(dp_hard_data)
        if ((m,"dags") in dp_data or (m,"dags") in dp_hard_data)
        and ((m,"nodags") in dp_data or (m,"nodags") in dp_hard_data)
    })
    if not all_models: st.info("No models have both dags and nodags runs to compare."); return

    def _dp_vals(model,mode,key):
        data=dp_data.get((model,mode)) or dp_hard_data.get((model,mode),[])
        return [r[key] for r in data if r.get(key) is not None]

    metric_labels,dags_vals,nodags_vals=[],[],[]
    for model in all_models:
        short=MODEL_LABELS.get(model,model)
        for pk,pl in [("pub_percentile","pub pct"),("priv_percentile","priv pct")]:
            d=_dp_vals(model,"dags",pk); n=_dp_vals(model,"nodags",pk)
            metric_labels.append(f"{short}\nDP {pl}")
            dags_vals.append(np.mean(d) if d else 0)
            nodags_vals.append(np.mean(n) if n else 0)
        bd=[r["exact_match"] for r in bio_data.get((model,"dags"),[])   if r.get("exact_match") is not None]
        bn=[r["exact_match"] for r in bio_data.get((model,"nodags"),[ ]) if r.get("exact_match") is not None]
        if bd or bn:
            metric_labels.append(f"{short}\nDSBio EM (%)")
            dags_vals.append(100*np.mean(bd) if bd else 0)
            nodags_vals.append(100*np.mean(bn) if bn else 0)

    x=np.arange(len(metric_labels)); w=0.35
    fig,axes=plt.subplots(2,2,figsize=(max(22, len(metric_labels)*1.5), 15))
    fig.suptitle("DAG vs No-DAG: Cross-Benchmark Overview",fontsize=13,fontweight="bold")

    ax=axes[0,0]
    b1=ax.bar(x-w/2,dags_vals,  w,label="DAGs",   color=COLORS["dags"],  alpha=0.85)
    b2=ax.bar(x+w/2,nodags_vals,w,label="No-DAGs",color=COLORS["nodags"],alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels,fontsize=7,rotation=90, ha="right")
    ax.set_ylabel("Metric value"); ax.set_title("Key Metrics: DAG vs No-DAG"); ax.legend()

    ax=axes[0,1]
    deltas=[d-n for d,n in zip(dags_vals,nodags_vals)]
    ax.barh(range(len(metric_labels)),deltas,
            color=["#2196F3" if d>=0 else "#F44336" for d in deltas],alpha=0.8)
    ax.set_yticks(range(len(metric_labels))); ax.set_yticklabels(metric_labels,fontsize=7)
    ax.set_xlabel("Δ = DAGs − No-DAGs"); ax.set_title("Performance Delta"); ax.axvline(0,color="black",lw=1)

    medal_dags,medal_nodags=[],[]
    for model in all_models:
        for mode,lst in [("dags",medal_dags),("nodags",medal_nodags)]:
            data=dp_data.get((model,mode)) or dp_hard_data.get((model,mode),[])
            sub=[r for r in data if "COMPLETE" in str(r.get("status",""))]
            lst.append(100*len([r for r in sub if r.get("public_medal") not in (None,"none")])/max(1,len(data)))
    x2=np.arange(len(all_models))
    ax=axes[1,0]
    b1=ax.bar(x2-w/2,medal_dags,  w,label="DAGs",   color=COLORS["dags"],  alpha=0.85)
    b2=ax.bar(x2+w/2,medal_nodags,w,label="No-DAGs",color=COLORS["nodags"],alpha=0.85)
    ax.set_xticks(x2); ax.set_xticklabels([MODEL_LABELS.get(m,m) for m in all_models],rotation=90, ha="right")
    ax.set_ylabel("Medal Rate (%)"); ax.set_title("Medal Rate (medals / all tasks)"); ax.legend()

    ax=axes[1,1]
    medal_deltas=[d-n for d,n in zip(medal_dags,medal_nodags)]
    ax.barh(range(len(all_models)),medal_deltas,
            color=["#2196F3" if d>=0 else "#F44336" for d in medal_deltas],alpha=0.8)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels([f"{MODEL_LABELS.get(m,m)}\nMedal Δ" for m in all_models],fontsize=8)
    ax.set_xlabel("Δ = DAGs − No-DAGs"); ax.set_title("Medal Rate Delta"); ax.axvline(0,color="black",lw=1)

    plt.tight_layout(); fig_to_st(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Trace Viewer
# ──────────────────────────────────────────────────────────────────────────────
def _render_turn(turn_idx, turn, show_dags, show_code, show_results, show_reasoning, search_term):
    """Render a single trajectory turn as a readable card."""
    role    = turn.get("role", "unknown")
    content = turn.get("content", "")
    turn_n  = turn.get("turn", turn_idx)
    elapsed = turn.get("step_time")
    reward  = turn.get("reward")

    if not content:
        return

    # ── Skip filter: if none of the requested content types exist in this turn ──
    has_dag      = '"dag_type"' in content
    has_python   = "<python>" in content
    has_result   = "<information>" in content or "<error>" in content
    has_reasoning= "<reasoning>" in content

    wanted = (
        (show_dags      and has_dag)
        or (show_code   and has_python)
        or (show_results and has_result and role == "user")
        or (show_reasoning and has_reasoning)
        or (role == "user" and not has_result)   # plain user message always shown
        or (has_dag)  # always show turns with DAGs so graph is visible
    )
    if not wanted:
        return

    # ── Search filter ──
    if search_term and search_term.lower() not in content.lower():
        return

    # Card colours
    bg   = "#1e3a5f" if role == "assistant" else "#2d2d2d"
    label = "🤖 Agent" if role == "assistant" else "🖥️ Environment"
    meta_parts = [f"Turn {turn_n}"]
    if elapsed is not None: meta_parts.append(f"{elapsed:.1f}s")
    if reward  is not None: meta_parts.append(f"reward={reward}")
    meta = " · ".join(meta_parts)

    st.markdown(
        f'<div style="background:{bg};border-radius:8px;padding:10px 14px 4px 14px;'
        f'margin-bottom:4px"><strong>{label}</strong> '
        f'<span style="color:#aaa;font-size:0.8em">{meta}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Parse and render content blocks ──
    # We split on known XML-style tags so each block gets its own widget
    remaining = content
    block_idx = 0

    def _pop_tag(text, tag):
        """Return (before, inner, after) for first occurrence of <tag>…</tag>, or None."""
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        m = pattern.search(text)
        if not m:
            return None
        return text[:m.start()], m.group(1), text[m.end():]

    while remaining.strip():
        block_idx += 1
        consumed = False

        # Try each structured block type in priority order
        for tag, label_text, show_flag, expander_open in [
            ("reasoning", "💭 Reasoning",   show_reasoning, True),
            ("python",    "🐍 Code",         show_code,      True),
            ("information","📤 Output",      show_results,   True),
            ("error",     "❌ Error output", show_results,   True),
        ]:
            result = _pop_tag(remaining, tag)
            if result is None:
                continue
            before, inner, after = result

            # Render any plain text before this tag
            if before.strip():
                _render_plain(before.strip(), has_dag, show_dags)

            if show_flag:
                with st.expander(f"{label_text}", expanded=expander_open):
                    if tag == "python":
                        st.code(inner.strip(), language="python")
                    elif tag in ("information", "error"):
                        # Truncate very long outputs for readability
                        txt = inner.strip()
                        if len(txt) > 4000:
                            st.text(txt[:4000])
                            st.caption(f"… {len(txt)-4000} more characters truncated")
                        else:
                            st.text(txt)
                    else:
                        st.markdown(inner.strip())

            remaining = after
            consumed = True
            break

        if not consumed:
            # No more structured tags — render remaining as plain text / DAG JSON
            _render_plain(remaining.strip(), has_dag, show_dags)
            break

    st.markdown("<hr style='margin:6px 0;border-color:#444'>", unsafe_allow_html=True)


def _render_dag_block(obj, show_dags):
    """Render a parsed DAG dict: graph plot + optional raw JSON."""
    dag_type = obj.get("dag_type", "")
    n_nodes  = len(obj.get("nodes", []))
    n_edges  = len(obj.get("edges", []))
    label    = f"🗂️ DAG — {dag_type} ({n_nodes} nodes, {n_edges} edges)"
    with st.expander(label, expanded=True):
        if n_nodes > 0:
            fig_w = max(8, min(14, n_nodes * 1.2))
            fig_h = max(5, min(10, n_nodes * 0.8))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            draw_dag_on_ax(obj, ax, f"{dag_type} — {n_nodes} nodes")
            plt.tight_layout()
            fig_to_st(fig)
        else:
            st.info("(empty DAG)")
        if show_dags:
            with st.expander("Raw JSON", expanded=False):
                st.json(obj)


def _render_plain(text, has_dag, show_dags):
    """Render plain text, detecting embedded DAG JSON (closed fence, unclosed fence, or bare)."""
    if not text:
        return

    # Collect all complete DAG objects from the text using the robust extractor
    dag_objs = []
    seen_keys = set()

    def _collect(candidate):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "dag_type" in obj:
                key = json.dumps(obj, sort_keys=True)
                if key not in seen_keys:
                    seen_keys.add(key)
                    dag_objs.append(obj)
        except Exception:
            pass

    # Closed fences
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        for c in _find_json_objects(m.group(1)):
            _collect(c)

    # Unclosed fence (DeepSeek)
    m = re.search(r"```(?:json)?\s*([\s\S]+)$", text)
    if m and "```" not in m.group(1):
        for c in _find_json_objects(m.group(1)):
            _collect(c)

    # Bare JSON
    for c in _find_json_objects(text):
        _collect(c)

    if dag_objs:
        # Render each DAG as a graph
        for obj in dag_objs:
            _render_dag_block(obj, show_dags)
        # Render any surrounding non-JSON prose
        # Strip out all JSON blobs and fence blocks before displaying as markdown
        prose = re.sub(r"```(?:json)?[\s\S]*?```", "", text)   # closed fences
        prose = re.sub(r"```(?:json)?[\s\S]+$", "", prose)      # unclosed fences
        prose = prose.strip()
        if prose:
            st.markdown(prose)
    else:
        st.markdown(text)


def page_trace_viewer(dp_data, dp_hard_data, bio_data):
    st.header("📜 Trace Viewer")
    st.caption("Read the full structured conversation trajectory for any run.")

    inv = {v:k for k,v in MODEL_LABELS.items()}

    # ── Dataset + run selection ────────────────────────────────────────────────
    col_ds, col_model, col_mode, col_task = st.columns([1,1,1,2])

    with col_ds:
        dataset = st.selectbox("Dataset", ["DSPredict Sampled","DSPredict Hard"], key="tv_ds")

    src = dp_data if dataset == "DSPredict Sampled" else dp_hard_data

    all_keys = list(src.keys())
    run_labels = [f"{MODEL_LABELS.get(m,m)} / {md}" for m,md in all_keys]

    with col_model:
        sel_run_label = st.selectbox("Run (model / mode)", run_labels, key="tv_run")
    sel_run_key = all_keys[run_labels.index(sel_run_label)]
    run_data = src.get(sel_run_key, [])

    # Build challenge list for this run
    challenges = [r.get("challenge","?") or f"sample_{r.get('sample_id','?')}" for r in run_data]
    with col_task:
        sel_ch = st.selectbox("Challenge / sample", challenges, key="tv_ch")

    row = next((r for r in run_data if (r.get("challenge") or f"sample_{r.get('sample_id','?')}") == sel_ch), None)
    if row is None:
        st.info("No data for this selection."); return

    trajectory = row.get("trajectory", [])
    if not trajectory:
        st.warning("This run has no trajectory stored."); return

    # ── Run metadata banner ───────────────────────────────────────────────────
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Status", str(row.get("status","?"))[:30])
    m2.metric("Turns",  str(row.get("total_turns","?")))
    time_s = row.get("execution_time")
    m3.metric("Time",   f"{time_s/60:.1f} min" if time_s else "—")
    m4.metric("Public medal",  str(row.get("public_medal","—")))
    m5.metric("Pub percentile",f"{row.get('pub_percentile','—')}")

    # ── Display filters ───────────────────────────────────────────────────────
    st.markdown("---")
    fc1,fc2,fc3,fc4,fc5 = st.columns([1,1,1,1,2])
    show_reasoning = fc1.checkbox("💭 Reasoning",  value=True,  key="tv_reasoning")
    show_code      = fc2.checkbox("🐍 Code",        value=True,  key="tv_code")
    show_results   = fc3.checkbox("📤 Outputs",     value=True,  key="tv_results")
    show_dags      = fc4.checkbox("🗂️ Show raw DAG JSON", value=False, key="tv_dags")
    search_term    = fc5.text_input("🔍 Search in content", placeholder="filter turns…", key="tv_search")

    # ── Turn range slider ──────────────────────────────────────────────────────
    n_turns = len(trajectory)
    if n_turns > 1:
        t_range = st.slider("Turn range", 1, n_turns, (1, n_turns), key="tv_range")
        turns_to_show = trajectory[t_range[0]-1 : t_range[1]]
        start_idx = t_range[0]-1
    else:
        turns_to_show = trajectory
        start_idx = 0

    st.caption(f"Showing {len(turns_to_show)} of {n_turns} turns")
    st.markdown("---")

    # ── Render turns ──────────────────────────────────────────────────────────
    for i, turn in enumerate(turns_to_show):
        _render_turn(start_idx + i, turn, show_dags, show_code, show_results, show_reasoning, search_term)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="DSGym Dashboard",page_icon="📊",
                       layout="wide",initial_sidebar_state="expanded")
    st.title("📊 DSGym Results Dashboard")
    st.caption("Interactive analysis of DAG vs No-DAG agents across DSPredict and DSBio benchmarks.")

    with st.spinner("Loading data…"):
        dp_data,dp_hard_data,bio_data=load_all_data()
    with st.spinner("Loading DAG consistency records…"):
        dag_records=load_dag_consistency_records()

    pages=["📋 Overview","🏆 DSPredict Analysis","🧬 DSBio Analysis",
           "🔀 DAG Visualizer","🔬 DAG Consistency","⚡ Cross-Benchmark",
           "📜 Trace Viewer"]
    page=st.sidebar.radio("Navigate",pages)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data loaded**")
    st.sidebar.markdown(f"- DSPredict Sampled: **{len(dp_data)}** runs")
    st.sidebar.markdown(f"- DSPredict Hard: **{len(dp_hard_data)}** runs")
    st.sidebar.markdown(f"- DSBio: **{len(bio_data)}** runs")
    st.sidebar.markdown(f"- DAG consistency files: **{len(dag_records)}**")

    if   page=="📋 Overview":          page_overview(dp_data,dp_hard_data,bio_data)
    elif page=="🏆 DSPredict Analysis": page_dspredict(dp_data,dp_hard_data)
    elif page=="🧬 DSBio Analysis":     page_dsbio(bio_data)
    elif page=="🔀 DAG Visualizer":     page_dag_visualizer(dp_data,dp_hard_data)
    elif page=="🔬 DAG Consistency":    page_dag_consistency(dag_records)
    elif page=="⚡ Cross-Benchmark":    page_cross_benchmark(dp_data,dp_hard_data,bio_data)
    elif page=="📜 Trace Viewer":       page_trace_viewer(dp_data,dp_hard_data,bio_data)


if __name__=="__main__":
    main()
