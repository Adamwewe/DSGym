"""
DAG Consistency Analysis metrics.

Measures:
1. Intra-model R→I consistency - how faithfully a model translates reasoning DAG into implementation DAG
2. Revision drift - how much DAGs change across plan revisions during execution
3. Code coverage - how well generated code covers planned implementation nodes

Supports multiple embedding backends:
- litellm (Fireworks AI qwen3-embedding-8b) - default, high quality
- sentence-transformers (fallback)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import numpy as np

from .base import BaseMetric, MetricResult

# Stop words for keyword extraction
STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "on", "from", "using",
    "use", "all", "each", "by", "into", "as", "at", "is", "are", "be", "its", "this", "that",
    "if", "any", "no", "data", "model", "step", "based", "new", "make", "get", "set", "run",
    "then", "also", "both", "only", "when", "will", "has", "have", "not", "their",
}


@dataclass
class DAGNode:
    """Represents a node in a DAG."""
    id: str
    label: str
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGNode":
        return cls(id=d.get("id", ""), label=d.get("label", ""))


@dataclass
class DAG:
    """Represents an extracted DAG."""
    dag_type: str  # 'reasoning', 'implementation', 'implementation_revision_1', etc.
    graph_id: str
    nodes: List[DAGNode]
    edges: List[Tuple[str, str]]  # (source_id, target_id)
    turn_index: int = 0  # Which turn in the trajectory this DAG came from
    raw_json: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], turn_index: int = 0) -> "DAG":
        nodes = [DAGNode.from_dict(n) for n in d.get("nodes", [])]
        edges = [(e["source"], e["target"]) for e in d.get("edges", []) if "source" in e and "target" in e]
        return cls(
            dag_type=d.get("dag_type", "unknown"),
            graph_id=d.get("graph_id", ""),
            nodes=nodes,
            edges=edges,
            turn_index=turn_index,
            raw_json=d,
        )
    
    @property
    def labels(self) -> List[str]:
        """Get all node labels (lowercase)."""
        return [n.label.lower() for n in self.nodes]
    
    @property
    def is_reasoning(self) -> bool:
        return self.dag_type == "reasoning"
    
    @property
    def is_implementation(self) -> bool:
        return "implementation" in self.dag_type
    
    @property
    def is_revision(self) -> bool:
        return "revision" in self.dag_type.lower()
    
    @property
    def revision_number(self) -> int:
        """Extract revision number from dag_type (e.g., 'implementation_revision_2' -> 2)."""
        if not self.is_revision:
            return 0
        match = re.search(r"revision[_-]?(\d+)", self.dag_type, re.IGNORECASE)
        return int(match.group(1)) if match else 1


@dataclass
class DAGSequence:
    """Represents all DAGs extracted from a trajectory."""
    dags: List[DAG]
    code_blocks: List[str]  # All <python>...</python> blocks from trajectory
    
    @property
    def reasoning_dags(self) -> List[DAG]:
        return [d for d in self.dags if d.is_reasoning]
    
    @property
    def implementation_dags(self) -> List[DAG]:
        return [d for d in self.dags if d.is_implementation]
    
    @property
    def n_revisions(self) -> int:
        """Count implementation revisions (excluding the initial implementation)."""
        return sum(1 for d in self.dags if d.is_revision)
    
    def get_dag_pairs(self) -> List[Tuple[DAG, DAG]]:
        """Get consecutive (prev, next) pairs of implementation DAGs for drift analysis."""
        impl_dags = sorted(self.implementation_dags, key=lambda d: (d.turn_index, d.revision_number))
        return list(zip(impl_dags[:-1], impl_dags[1:])) if len(impl_dags) > 1 else []
    
    def get_reasoning_implementation_pairs(self) -> List[Tuple[DAG, DAG]]:
        """Get (reasoning, implementation) pairs, matching by turn index proximity."""
        pairs = []
        r_dags = self.reasoning_dags
        i_dags = [d for d in self.implementation_dags if not d.is_revision]
        
        for r_dag in r_dags:
            # Find closest implementation DAG after this reasoning DAG
            candidates = [d for d in i_dags if d.turn_index >= r_dag.turn_index]
            if candidates:
                pairs.append((r_dag, min(candidates, key=lambda d: d.turn_index)))
        
        return pairs


def _find_json_objects(text: str) -> List[str]:
    """
    Scan text and return all top-level JSON object strings (balanced braces).
    Handles nested objects and strings with escaped characters correctly.
    """
    results = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                results.append(text[start : i + 1])
                start = -1
    return results


def extract_dags_from_content(content: str, turn_index: int = 0) -> List[DAG]:
    """
    Extract all DAG objects from a single message content.
    
    Args:
        content: The message content (assistant response)
        turn_index: Index of the turn in the trajectory
        
    Returns:
        List of DAG objects, in appearance order, deduplicated by content.
    """
    dags: List[DAG] = []
    seen: Set[str] = set()

    def _add(candidate: str) -> None:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "dag_type" in obj:
                key = json.dumps(obj, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    dags.append(DAG.from_dict(obj, turn_index))
        except (json.JSONDecodeError, ValueError):
            pass

    # 1. Code fences first (model often wraps JSON in ```json ... ```)
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", content):
        for candidate in _find_json_objects(m.group(1)):
            _add(candidate)

    # 2. Any remaining top-level JSON objects in the full text
    for candidate in _find_json_objects(content):
        _add(candidate)

    return dags


def extract_dags_from_trajectory(trajectory: List[Dict[str, Any]]) -> DAGSequence:
    """
    Extract all DAGs and code blocks from a full trajectory.
    
    Args:
        trajectory: List of turn dicts with 'role' and 'content' keys
        
    Returns:
        DAGSequence containing all DAGs and code blocks
    """
    all_dags: List[DAG] = []
    all_code: List[str] = []
    
    for turn_idx, turn in enumerate(trajectory):
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "assistant":
            continue
            
        content = turn.get("content", "")
        
        # Extract DAGs
        dags = extract_dags_from_content(content, turn_idx)
        all_dags.extend(dags)
        
        # Extract code blocks
        code_blocks = re.findall(r"<python>(.*?)</python>", content, re.DOTALL)
        all_code.extend(code_blocks)
    
    return DAGSequence(dags=all_dags, code_blocks=all_code)


def keywords_from_labels(labels: List[str]) -> Set[str]:
    """Extract meaningful keywords from a list of node labels."""
    words = set()
    for lbl in labels:
        for w in re.findall(r"[a-z][a-z0-9]+", lbl.lower()):
            if w not in STOP_WORDS and len(w) > 2:
                words.add(w)
    return words


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def rouge_scores(labels_a: List[str], labels_b: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-1/2/L F1 between two ordered lists of node labels.
    
    Args:
        labels_a: First list of node labels
        labels_b: Second list of node labels
        
    Returns:
        Dict with rouge1, rouge2, rougeL scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    a = " ".join(labels_a) if labels_a else ""
    b = " ".join(labels_b) if labels_b else ""
    
    if not a or not b:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    s = scorer.score(a, b)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }


class DAGEmbedder:
    """
    Computes embeddings for DAG node labels using litellm or sentence-transformers.
    
    Default: Fireworks AI qwen3-embedding-8b via litellm (high quality, requires API key)
    Fallback: sentence-transformers all-MiniLM-L6-v2 (local, lower quality)
    
    Supports optional caching to avoid recalculating embeddings.
    Tracks API costs for litellm calls.
    """
    
    def __init__(
        self,
        model: str = "fireworks_ai/fireworks/qwen3-embedding-8b",
        fallback_model: str = "all-MiniLM-L6-v2",
        use_fallback: bool = False,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            model: litellm-compatible embedding model name
            fallback_model: sentence-transformers model name for fallback
            use_fallback: If True, use sentence-transformers instead of litellm
            cache_path: Optional path to pickle file for embedding cache
        """
        self.model = model
        self.fallback_model = fallback_model
        self.use_fallback = use_fallback
        self._st_model = None  # Lazy-loaded sentence-transformers model
        self.cache_path = cache_path
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_dirty = False
        
        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        
        # Load cache if exists
        if cache_path and os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
            except Exception:
                self._cache = {}
    
    def save_cache(self):
        """Save embedding cache to disk."""
        if self.cache_path and self._cache_dirty:
            import pickle
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
            self._cache_dirty = False
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of API usage and costs."""
        return {
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "model": self.model,
        }
    
    def embed_cached(self, key: str, texts: List[str]) -> np.ndarray:
        """
        Get embedding from cache or compute and cache it.
        
        Args:
            key: Unique cache key for this embedding
            texts: List of strings to embed
            
        Returns:
            Mean-pooled embedding vector
        """
        if key in self._cache:
            return self._cache[key]
        
        embeddings = self.embed(texts)
        if embeddings.size == 0:
            return np.array([])
        
        mean_emb = np.mean(embeddings, axis=0)
        self._cache[key] = mean_emb
        self._cache_dirty = True
        return mean_emb
    
    def embed_batch_async(self, batch_items: List[Tuple[str, List[str]]]) -> Dict[str, np.ndarray]:
        """
        Embed multiple items concurrently using async calls.
        
        Args:
            batch_items: List of (cache_key, texts) tuples
            
        Returns:
            Dict mapping cache_key to mean-pooled embedding
        """
        import asyncio
        
        # Filter out cached items
        uncached = [(key, texts) for key, texts in batch_items if key not in self._cache]
        results = {key: self._cache[key] for key, _ in batch_items if key in self._cache}
        
        if not uncached:
            return results
        
        # Collect all texts and track indices
        all_texts = []
        key_ranges = []  # (key, start_idx, end_idx)
        for key, texts in uncached:
            if texts and any(texts):
                start = len(all_texts)
                all_texts.extend(texts)
                key_ranges.append((key, start, len(all_texts)))
        
        if not all_texts:
            return results
        
        # Single batched API call (litellm handles batching efficiently)
        embeddings = self.embed(all_texts)
        
        # Split results back to keys and mean pool
        for key, start, end in key_ranges:
            emb_slice = embeddings[start:end]
            if emb_slice.size > 0:
                mean_emb = np.mean(emb_slice, axis=0)
                if np.linalg.norm(mean_emb) > 0 and not np.isnan(mean_emb).any():
                    self._cache[key] = mean_emb
                    self._cache_dirty = True
                    results[key] = mean_emb
        
        return results
        
    def _get_st_model(self):
        """Lazy load sentence-transformers model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.fallback_model)
        return self._st_model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self.use_fallback:
            return self._embed_st(texts)
        
        try:
            return self._embed_litellm(texts)
        except Exception as e:
            # Fallback to sentence-transformers
            import warnings
            warnings.warn(f"litellm embedding failed ({e}), falling back to sentence-transformers")
            return self._embed_st(texts)
    
    def _embed_litellm(self, texts: List[str]) -> np.ndarray:
        """Embed using litellm with cost tracking."""
        import litellm
        
        response = litellm.embedding(
            model=self.model,
            input=texts,
        )
        
        # Track usage and costs
        self.api_calls += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += getattr(response.usage, 'total_tokens', 0)
        
        # Try to get cost from litellm
        try:
            from litellm import completion_cost
            cost = completion_cost(completion_response=response)
            self.total_cost += cost
        except Exception:
            # Estimate cost if completion_cost fails (Fireworks embedding ~$0.016/1M tokens)
            tokens = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else len(texts) * 50
            self.total_cost += tokens * 0.000016  # $0.016 per 1M tokens
        
        embeddings = [item["embedding"] for item in response.data]
        return np.array(embeddings)
    
    def _embed_st(self, texts: List[str]) -> np.ndarray:
        """Embed using sentence-transformers."""
        model = self._get_st_model()
        return model.encode(texts)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def dag_similarity(self, dag_a: Union[DAG, Dict], dag_b: Union[DAG, Dict]) -> float:
        """
        Compute semantic similarity between two DAGs based on their node labels.
        
        Uses mean pooling of node embeddings.
        Accepts both DAG objects and raw dict representations.
        """
        # Handle dict inputs by extracting labels directly
        if isinstance(dag_a, dict):
            labels_a = [n.get("label", "").lower() for n in dag_a.get("nodes", [])]
        else:
            labels_a = dag_a.labels
            
        if isinstance(dag_b, dict):
            labels_b = [n.get("label", "").lower() for n in dag_b.get("nodes", [])]
        else:
            labels_b = dag_b.labels
        
        if not labels_a or not labels_b:
            return 0.0
        
        # Get embeddings for all labels
        all_labels = labels_a + labels_b
        embeddings = self.embed(all_labels)
        
        if embeddings.size == 0:
            return 0.0
        
        emb_a = embeddings[:len(labels_a)]
        emb_b = embeddings[len(labels_a):]
        
        # Mean pooling
        mean_a = np.mean(emb_a, axis=0)
        mean_b = np.mean(emb_b, axis=0)
        
        return self.cosine_similarity(mean_a, mean_b)


@dataclass
class DAGPairMetrics:
    """Metrics comparing two DAGs."""
    dag_a_type: str
    dag_b_type: str
    dag_a_turn: int
    dag_b_turn: int
    
    # Node counts
    n_nodes_a: int
    n_nodes_b: int
    node_delta: int  # b - a
    
    # Keyword-based metrics
    jaccard: float
    overlap: float  # fraction of A's keywords in B
    
    # ROUGE scores
    rouge1: float
    rouge2: float
    rougeL: float
    
    # Embedding similarity (optional)
    embedding_similarity: Optional[float] = None


@dataclass
class CodeCoverageMetrics:
    """Metrics for how well code covers the implementation DAG."""
    total_nodes: int
    covered_nodes: int
    coverage_ratio: float
    first_miss_position: Optional[float]  # Position (0-1) of first uncovered node
    uncovered_labels: List[str]


@dataclass
class DAGConsistencyResult:
    """Full DAG consistency analysis result."""
    # Basic info
    n_total_dags: int
    n_reasoning_dags: int
    n_implementation_dags: int
    n_revisions: int
    
    # R→I consistency (first reasoning to first implementation)
    ri_metrics: Optional[DAGPairMetrics] = None
    
    # Revision drift metrics (each consecutive pair of implementation DAGs)
    revision_metrics: List[DAGPairMetrics] = field(default_factory=list)
    
    # Code coverage
    code_coverage: Optional[CodeCoverageMetrics] = None
    
    # All R→I pairs (for multi-plan cases)
    all_ri_metrics: List[DAGPairMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "n_total_dags": self.n_total_dags,
            "n_reasoning_dags": self.n_reasoning_dags,
            "n_implementation_dags": self.n_implementation_dags,
            "n_revisions": self.n_revisions,
        }
        
        if self.ri_metrics:
            result["ri_metrics"] = {
                "jaccard": self.ri_metrics.jaccard,
                "overlap": self.ri_metrics.overlap,
                "rouge1": self.ri_metrics.rouge1,
                "rouge2": self.ri_metrics.rouge2,
                "rougeL": self.ri_metrics.rougeL,
                "node_delta": self.ri_metrics.node_delta,
                "embedding_similarity": self.ri_metrics.embedding_similarity,
            }
        
        if self.revision_metrics:
            result["revision_drift"] = [
                {
                    "from_type": m.dag_a_type,
                    "to_type": m.dag_b_type,
                    "from_turn": m.dag_a_turn,
                    "to_turn": m.dag_b_turn,
                    "jaccard": m.jaccard,
                    "overlap": m.overlap,
                    "rouge1": m.rouge1,
                    "rouge2": m.rouge2,
                    "rougeL": m.rougeL,
                    "node_delta": m.node_delta,
                    "embedding_similarity": m.embedding_similarity,
                }
                for m in self.revision_metrics
            ]
        
        if self.code_coverage:
            result["code_coverage"] = {
                "total_nodes": self.code_coverage.total_nodes,
                "covered_nodes": self.code_coverage.covered_nodes,
                "coverage_ratio": self.code_coverage.coverage_ratio,
                "first_miss_position": self.code_coverage.first_miss_position,
            }
        
        return result


def compute_dag_pair_metrics(
    dag_a: DAG,
    dag_b: DAG,
    embedder: Optional[DAGEmbedder] = None,
) -> DAGPairMetrics:
    """
    Compute all comparison metrics between two DAGs.
    
    Args:
        dag_a: First DAG (e.g., reasoning or previous implementation)
        dag_b: Second DAG (e.g., implementation or revised implementation)
        embedder: Optional DAGEmbedder for semantic similarity
        
    Returns:
        DAGPairMetrics object
    """
    labels_a = dag_a.labels
    labels_b = dag_b.labels
    
    kw_a = keywords_from_labels(labels_a)
    kw_b = keywords_from_labels(labels_b)
    
    # Jaccard and overlap
    jacc = jaccard_similarity(kw_a, kw_b)
    overlap = len(kw_a & kw_b) / max(1, len(kw_a)) if kw_a else 0.0
    
    # ROUGE
    rouge = rouge_scores(labels_a, labels_b)
    
    # Embedding similarity (optional)
    emb_sim = None
    if embedder:
        try:
            emb_sim = embedder.dag_similarity(dag_a, dag_b)
        except Exception:
            pass
    
    return DAGPairMetrics(
        dag_a_type=dag_a.dag_type,
        dag_b_type=dag_b.dag_type,
        dag_a_turn=dag_a.turn_index,
        dag_b_turn=dag_b.turn_index,
        n_nodes_a=len(labels_a),
        n_nodes_b=len(labels_b),
        node_delta=len(labels_b) - len(labels_a),
        jaccard=jacc,
        overlap=overlap,
        rouge1=rouge["rouge1"],
        rouge2=rouge["rouge2"],
        rougeL=rouge["rougeL"],
        embedding_similarity=emb_sim,
    )


def compute_code_coverage(
    dag: DAG,
    code_blocks: List[str],
) -> CodeCoverageMetrics:
    """
    Compute how well the code covers the implementation DAG nodes.
    
    Args:
        dag: Implementation DAG
        code_blocks: List of code block strings
        
    Returns:
        CodeCoverageMetrics object
    """
    all_code = " ".join(code_blocks).lower()
    code_words = set(re.findall(r"[a-z][a-z0-9]+", all_code))
    
    labels = dag.labels
    hits = []
    uncovered = []
    
    for lbl in labels:
        lbl_kw = keywords_from_labels([lbl])
        hit = bool(lbl_kw & code_words)
        hits.append(hit)
        if not hit:
            uncovered.append(lbl)
    
    total = len(labels)
    covered = sum(hits)
    coverage = covered / max(1, total)
    
    # Find position of first miss
    first_miss = None
    for i, h in enumerate(hits):
        if not h:
            first_miss = i / max(1, len(hits))
            break
    
    return CodeCoverageMetrics(
        total_nodes=total,
        covered_nodes=covered,
        coverage_ratio=coverage,
        first_miss_position=first_miss,
        uncovered_labels=uncovered,
    )


def analyze_dag_consistency(
    trajectory: List[Dict[str, Any]],
    compute_embeddings: bool = True,
    embedding_model: str = "fireworks_ai/fireworks/qwen3-embedding-8b",
    use_fallback_embeddings: bool = False,
) -> DAGConsistencyResult:
    """
    Perform full DAG consistency analysis on a trajectory.
    
    Args:
        trajectory: List of turn dicts from agent execution
        compute_embeddings: Whether to compute embedding similarities
        embedding_model: litellm embedding model to use
        use_fallback_embeddings: Use sentence-transformers instead of litellm
        
    Returns:
        DAGConsistencyResult with all metrics
    """
    # Extract all DAGs and code
    dag_seq = extract_dags_from_trajectory(trajectory)
    
    # Initialize embedder if needed
    embedder = None
    if compute_embeddings:
        embedder = DAGEmbedder(
            model=embedding_model,
            use_fallback=use_fallback_embeddings,
        )
    
    # Basic counts
    result = DAGConsistencyResult(
        n_total_dags=len(dag_seq.dags),
        n_reasoning_dags=len(dag_seq.reasoning_dags),
        n_implementation_dags=len(dag_seq.implementation_dags),
        n_revisions=dag_seq.n_revisions,
    )
    
    # R→I consistency (all pairs)
    ri_pairs = dag_seq.get_reasoning_implementation_pairs()
    for r_dag, i_dag in ri_pairs:
        metrics = compute_dag_pair_metrics(r_dag, i_dag, embedder)
        result.all_ri_metrics.append(metrics)
    
    # Primary R→I metrics (first pair)
    if result.all_ri_metrics:
        result.ri_metrics = result.all_ri_metrics[0]
    
    # Revision drift (consecutive implementation pairs)
    impl_pairs = dag_seq.get_dag_pairs()
    for prev_dag, next_dag in impl_pairs:
        metrics = compute_dag_pair_metrics(prev_dag, next_dag, embedder)
        result.revision_metrics.append(metrics)
    
    # Code coverage (using last implementation DAG if available)
    impl_dags = dag_seq.implementation_dags
    if impl_dags:
        last_impl = max(impl_dags, key=lambda d: (d.turn_index, d.revision_number))
        result.code_coverage = compute_code_coverage(last_impl, dag_seq.code_blocks)
    
    return result


class DAGConsistencyMetric(BaseMetric):
    """
    Metric wrapper for DAG consistency analysis.
    
    Computes R→I consistency, revision drift, and code coverage metrics.
    """
    
    def __init__(
        self,
        compute_embeddings: bool = True,
        embedding_model: str = "fireworks_ai/fireworks/qwen3-embedding-8b",
        use_fallback_embeddings: bool = False,
        **kwargs
    ):
        """
        Initialize DAG consistency metric.
        
        Args:
            compute_embeddings: Whether to compute embedding similarities
            embedding_model: litellm embedding model to use
            use_fallback_embeddings: Use sentence-transformers instead of litellm
        """
        super().__init__(**kwargs)
        self.compute_embeddings = compute_embeddings
        self.embedding_model = embedding_model
        self.use_fallback_embeddings = use_fallback_embeddings
    
    @property
    def name(self) -> str:
        return "dag_consistency"
    
    @property
    def requires_ground_truth(self) -> bool:
        return False
    
    def evaluate(
        self,
        prediction: str,
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate DAG consistency from trajectory in extra_info.
        
        Args:
            prediction: Not used (DAGs come from trajectory)
            ground_truth: Not used
            query: Not used
            extra_info: Must contain 'trajectory' key with list of turns
            
        Returns:
            MetricResult with DAG consistency metrics
        """
        extra_info = extra_info or {}
        trajectory = extra_info.get("trajectory", [])
        
        if not trajectory:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No trajectory available"}
            )
        
        try:
            result = analyze_dag_consistency(
                trajectory,
                compute_embeddings=self.compute_embeddings,
                embedding_model=self.embedding_model,
                use_fallback_embeddings=self.use_fallback_embeddings,
            )
            
            # Primary score is R→I jaccard (or code coverage if no R→I)
            score = None
            if result.ri_metrics:
                score = result.ri_metrics.jaccard
            elif result.code_coverage:
                score = result.code_coverage.coverage_ratio
            
            return MetricResult(
                metric_name=self.name,
                score=score,
                details=result.to_dict()
            )
            
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=None,
                error=str(e)
            )
