"""Semantic similarity matching using embeddings."""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Anchor, Config, Paper


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
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
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
        """Get embedding for text, using cache if available."""
        if cache_key and cache_key in self._embedding_cache:
            return np.array(self._embedding_cache[cache_key])
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if cache_key:
            self._embedding_cache[cache_key] = embedding.tolist()
            self._save_cache()
        
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
    
    def get_paper_embedding(self, paper: Paper) -> np.ndarray:
        """Get embedding for a paper."""
        text = f"{paper.title}\n\n{paper.abstract}"
        cache_key = self._get_cache_key(text, f"paper_{paper.id}")
        return self.embed_text(text, cache_key)
    
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
        
        # Get anchor embeddings
        anchor_embeddings = self.get_anchor_embeddings(anchors)
        
        # Score each paper
        scored_papers = []
        for paper in papers:
            score = self.score_paper(paper, anchor_embeddings)
            if score >= threshold:
                paper.relevance_score = score
                scored_papers.append(paper)
        
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
        
        scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored_papers[:max_results]

