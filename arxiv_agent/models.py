"""Data models for the arxiv agent."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Paper:
    """Represents an arxiv paper."""

    id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    link: str
    pdf_link: str | None = None
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "link": self.link,
            "pdf_link": self.pdf_link,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        return cls(
            id=data["id"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data["authors"],
            categories=data["categories"],
            published=datetime.fromisoformat(data["published"]),
            updated=datetime.fromisoformat(data["updated"]),
            link=data["link"],
            pdf_link=data.get("pdf_link"),
        )


@dataclass
class Anchor:
    """Represents an interest anchor - either a topic or a saved paper."""

    id: str
    type: str  # "topic" or "paper"
    text: str  # The topic description or paper title + abstract
    title: str  # Display title
    added_at: datetime = field(default_factory=datetime.now)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "title": self.title,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Anchor":
        return cls(
            id=data["id"],
            type=data["type"],
            text=data["text"],
            title=data["title"],
            added_at=datetime.fromisoformat(data["added_at"]),
        )


# Available sentence embedding models with descriptions
EMBEDDING_MODELS: dict[str, str] = {
    "all-MiniLM-L6-v2": "Fast & lightweight (default)",
    "all-MiniLM-L12-v2": "Better quality, slightly slower",
    "all-mpnet-base-v2": "Best quality, slower",
    "multi-qa-MiniLM-L6-cos-v1": "Optimized for search/QA",
    "paraphrase-MiniLM-L6-v2": "Good for paraphrasing",
    "all-distilroberta-v1": "DistilRoBERTa based",
}


# Available arxiv categories with descriptions
ARXIV_CATEGORIES: dict[str, str] = {
    # Computer Science
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.IR": "Information Retrieval",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.HC": "Human-Computer Interaction",
    "cs.SE": "Software Engineering",
    "cs.DC": "Distributed Computing",
    "cs.CR": "Cryptography and Security",
    "cs.DB": "Databases",
    "cs.PL": "Programming Languages",
    # Statistics
    "stat.ML": "Machine Learning (Stats)",
    "stat.TH": "Statistics Theory",
    # Math
    "math.OC": "Optimization and Control",
    "math.ST": "Statistics Theory",
    # Electrical Engineering
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    # Quantitative Biology
    "q-bio.NC": "Neurons and Cognition",
    # Physics
    "physics.comp-ph": "Computational Physics",
}


@dataclass
class Config:
    """Agent configuration."""

    data_dir: str = "~/.arxiv_agent"
    embedding_model: str = "all-MiniLM-L6-v2"
    relevance_threshold: float = 0.35
    max_results: int = 50

    # Default categories to monitor
    categories: list[str] = field(
        default_factory=lambda: [
            "cs.LG",  # Machine Learning
            "cs.AI",  # Artificial Intelligence
            "cs.CL",  # Computation and Language
            "cs.IR",  # Information Retrieval
        ]
    )

    @property
    def feeds(self) -> list[str]:
        """Generate RSS feed URLs from categories."""
        return [f"https://rss.arxiv.org/rss/{cat}" for cat in self.categories]
