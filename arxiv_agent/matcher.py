"""Semantic similarity matching using embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .models import Anchor, Config, Paper

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _hf_model_is_cached(model_name: str) -> bool:
    """Return True if the embedding model is already in the local HF cache.

    Checked via the filesystem (without importing huggingface_hub) so the
    caller can decide whether to enable offline mode before that import.
    """
    # Bare names like "all-MiniLM-L6-v2" live under the sentence-transformers org.
    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        "hub",
    )
    snapshots = Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


class SemanticMatcher:
    """Matches papers to anchors using semantic similarity."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(os.path.expanduser(config.data_dir))
        # Use model-specific cache file to avoid dimension mismatches
        model_name = config.embedding_model.replace("/", "_").replace("\\", "_")
        self.cache_file = self.data_dir / f"embeddings_cache_{model_name}.json"
        self._model: SentenceTransformer | None = None
        self._embedding_cache: dict[str, list[float]] = {}
        self._load_cache()
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model.

        The ``sentence_transformers`` import lives here (not at module top) so
        that merely importing this module does not pull in torch. Only code
        paths that actually embed text pay that ~400MB cost. The web server
        relies on this to stay torch-free and run embedding in a subprocess
        (see ``run_matcher_in_subprocess``).
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # If the model is already in the HF cache, load it purely from local
            # files. This skips the unauthenticated "check for updates" request
            # to the Hub on every cold start (which prints a HF_TOKEN warning and
            # adds a network round-trip). First-time downloads and switching to a
            # new model still work, since we only do this when it's cached.
            self._model = SentenceTransformer(
                self.config.embedding_model,
                local_files_only=_hf_model_is_cached(self.config.embedding_model),
            )
        return self._model
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._embedding_cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._embedding_cache = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._embedding_cache, f)
    
    def _get_cache_key(self, text: str, prefix: str = "") -> str:
        """Generate a cache key for text."""
        # Use hash of text for the key
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{prefix}_{text_hash}" if prefix else text_hash
    
    def embed_text(self, text: str, cache_key: str | None = None) -> np.ndarray:
        """Get embedding for text, using cache if available.

        Updates the in-memory cache but does NOT persist to disk. Callers that
        do batch work should call `_save_cache()` once at the end.
        """
        if cache_key and cache_key in self._embedding_cache:
            return np.array(self._embedding_cache[cache_key])

        embedding = self.model.encode(text, convert_to_numpy=True)

        if cache_key:
            self._embedding_cache[cache_key] = embedding.tolist()

        return embedding
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batch embed multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get_anchor_embeddings(self, anchors: list[Anchor]) -> np.ndarray:
        """Get embeddings for all anchors."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, anchor in enumerate(anchors):
            cache_key = self._get_cache_key(anchor.text, f"anchor_{anchor.id}")
            if cache_key in self._embedding_cache:
                embeddings.append(np.array(self._embedding_cache[cache_key]))
            else:
                embeddings.append(None)
                texts_to_embed.append(anchor.text)
                indices_to_embed.append(i)
        
        # Batch embed missing texts
        if texts_to_embed:
            new_embeddings = self.embed_texts(texts_to_embed)
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                embeddings[idx] = emb
                cache_key = self._get_cache_key(text, f"anchor_{anchors[idx].id}")
                self._embedding_cache[cache_key] = emb.tolist()
            self._save_cache()
        
        return np.array(embeddings)
    
    def _paper_text_and_key(self, paper: Paper) -> tuple[str, str]:
        text = f"{paper.title}\n\n{paper.abstract}"
        return text, self._get_cache_key(text, f"paper_{paper.id}")

    def get_paper_embedding(self, paper: Paper) -> np.ndarray:
        """Get embedding for a paper."""
        text, cache_key = self._paper_text_and_key(paper)
        return self.embed_text(text, cache_key)

    def _prefill_paper_cache(self, papers: list[Paper]) -> bool:
        """Batch-encode any papers not yet in the cache. Returns True if cache changed."""
        texts: list[str] = []
        keys: list[str] = []
        seen: set[str] = set()
        for paper in papers:
            text, key = self._paper_text_and_key(paper)
            if key in self._embedding_cache or key in seen:
                continue
            seen.add(key)
            texts.append(text)
            keys.append(key)
        if not texts:
            return False
        embeddings = self.embed_texts(texts)
        for key, emb in zip(keys, embeddings):
            self._embedding_cache[key] = emb.tolist()
        return True
    
    def score_paper(self, paper: Paper, anchor_embeddings: np.ndarray) -> float:
        """
        Score a paper's relevance to the anchors.
        Returns the maximum similarity across all anchors.
        """
        if len(anchor_embeddings) == 0:
            return 0.0
        
        paper_embedding = self.get_paper_embedding(paper)
        
        # Compute similarity to each anchor
        similarities = []
        for anchor_emb in anchor_embeddings:
            sim = self.cosine_similarity(paper_embedding, anchor_emb)
            similarities.append(sim)
        
        # Return max similarity (paper is relevant if it matches any anchor well)
        return max(similarities)
    
    def filter_papers(
        self,
        papers: list[Paper],
        anchors: list[Anchor],
        threshold: float | None = None,
        max_results: int | None = None,
    ) -> list[Paper]:
        """
        Filter and rank papers by relevance to anchors.
        
        Args:
            papers: List of papers to filter
            anchors: List of interest anchors
            threshold: Minimum relevance score (default: config.relevance_threshold)
            max_results: Maximum number of results (default: config.max_results)
        
        Returns:
            List of papers with relevance_score set, sorted by relevance
        """
        if not anchors:
            return []
        
        threshold = threshold if threshold is not None else self.config.relevance_threshold
        max_results = max_results if max_results is not None else self.config.max_results
        
        # Get anchor embeddings (this already persists the cache if anchors were added)
        anchor_embeddings = self.get_anchor_embeddings(anchors)

        # Batch-encode any uncached papers in a single model.encode() call,
        # so the score loop below only ever hits the in-memory cache.
        cache_dirty = self._prefill_paper_cache(papers)

        # Score each paper
        scored_papers = []
        for paper in papers:
            score = self.score_paper(paper, anchor_embeddings)
            if score >= threshold:
                paper.relevance_score = score
                scored_papers.append(paper)

        if cache_dirty:
            self._save_cache()

        # Sort by relevance score, highest first
        scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        return scored_papers[:max_results]
    
    def find_similar_papers(
        self,
        reference_paper: Paper,
        papers: list[Paper],
        threshold: float = 0.5,
        max_results: int = 10,
    ) -> list[Paper]:
        """Find papers similar to a reference paper."""
        cache_dirty = self._prefill_paper_cache([reference_paper] + papers)
        ref_embedding = self.get_paper_embedding(reference_paper)

        scored_papers = []
        for paper in papers:
            if paper.id == reference_paper.id:
                continue

            paper_embedding = self.get_paper_embedding(paper)
            score = self.cosine_similarity(ref_embedding, paper_embedding)

            if score >= threshold:
                paper.relevance_score = score
                scored_papers.append(paper)

        if cache_dirty:
            self._save_cache()

        scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored_papers[:max_results]


def _subprocess_entry(queue, config: Config, method: str, payload: tuple) -> None:
    """Run a SemanticMatcher method inside a freshly spawned child process.

    This is the target of the spawned process: it constructs a matcher (which
    imports torch and loads the model on first use), does the work, ships the
    result back over the queue, and then the process exits — at which point the
    OS reclaims all of torch's memory.
    """
    try:
        matcher = SemanticMatcher(config)
        result = getattr(matcher, method)(*payload)
        queue.put(("ok", result))
    except Exception:  # pragma: no cover - surfaced to parent below
        import traceback

        queue.put(("error", traceback.format_exc()))


def run_matcher_in_subprocess(config: Config, method: str, *payload):
    """Run a ``SemanticMatcher`` method in a short-lived subprocess.

    The long-lived caller (e.g. the web server) never imports torch and stays
    small (~66MB). Each call spawns a child that loads the model, does the
    embedding/scoring, returns the result, and exits, fully releasing memory.

    Args:
        config: agent configuration (must be picklable).
        method: name of the SemanticMatcher method to call, e.g. "filter_papers".
        *payload: positional arguments forwarded to that method (must be
            picklable; Paper/Anchor dataclasses are).

    Returns:
        Whatever the underlying method returns (e.g. a list of scored Papers).
    """
    import multiprocessing
    import queue as _queue

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_entry, args=(result_queue, config, method, payload)
    )
    proc.start()

    # Wait for a result, but don't hang forever if the child dies (e.g. OOM kill
    # or segfault) before it can report back.
    while True:
        try:
            status, result = result_queue.get(timeout=2)
            break
        except _queue.Empty:
            if not proc.is_alive():
                proc.join()
                raise RuntimeError(
                    f"Embedding subprocess exited (code {proc.exitcode}) "
                    "before returning a result"
                )

    proc.join()
    if status == "error":
        raise RuntimeError(f"Embedding subprocess failed:\n{result}")
    return result

