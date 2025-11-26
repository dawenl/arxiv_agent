"""Anchor storage and management for research interests."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .models import Anchor, Config, Paper


class AnchorStore:
    """Manages storage and retrieval of interest anchors."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(os.path.expanduser(config.data_dir))
        self.anchors_file = self.data_dir / "anchors.json"
        self.cache_file = self.data_dir / "embeddings_cache.json"
        self._anchors: list[Anchor] = []
        self._ensure_data_dir()
        self._load()
    
    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> None:
        """Load anchors from disk."""
        if self.anchors_file.exists():
            try:
                with open(self.anchors_file, "r") as f:
                    data = json.load(f)
                self._anchors = [Anchor.from_dict(a) for a in data.get("anchors", [])]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load anchors: {e}")
                self._anchors = []
        else:
            self._anchors = []
    
    def _save(self) -> None:
        """Save anchors to disk."""
        data = {
            "anchors": [a.to_dict() for a in self._anchors],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.anchors_file, "w") as f:
            json.dump(data, f, indent=2)
    
    @property
    def anchors(self) -> list[Anchor]:
        """Get all anchors."""
        return self._anchors.copy()
    
    def add_topic(self, topic: str, title: str | None = None) -> Anchor:
        """Add a topic as an anchor."""
        anchor = Anchor(
            id=str(uuid.uuid4())[:8],
            type="topic",
            text=topic,
            title=title or topic[:50],
            added_at=datetime.now(),
        )
        self._anchors.append(anchor)
        self._save()
        return anchor
    
    def add_paper(self, paper: Paper) -> Anchor:
        """Add a paper as an anchor."""
        # Combine title and abstract for better matching
        text = f"{paper.title}\n\n{paper.abstract}"
        
        anchor = Anchor(
            id=paper.id,
            type="paper",
            text=text,
            title=paper.title,
            added_at=datetime.now(),
        )
        
        # Check if already exists
        existing = next((a for a in self._anchors if a.id == anchor.id), None)
        if existing:
            return existing
        
        self._anchors.append(anchor)
        self._save()
        return anchor
    
    def remove_anchor(self, anchor_id: str) -> bool:
        """Remove an anchor by ID."""
        original_len = len(self._anchors)
        self._anchors = [a for a in self._anchors if a.id != anchor_id]
        if len(self._anchors) < original_len:
            self._save()
            return True
        return False
    
    def get_anchor(self, anchor_id: str) -> Anchor | None:
        """Get an anchor by ID."""
        return next((a for a in self._anchors if a.id == anchor_id), None)
    
    def get_topics(self) -> list[Anchor]:
        """Get all topic anchors."""
        return [a for a in self._anchors if a.type == "topic"]
    
    def get_papers(self) -> list[Anchor]:
        """Get all paper anchors."""
        return [a for a in self._anchors if a.type == "paper"]
    
    def clear_all(self) -> None:
        """Remove all anchors."""
        self._anchors = []
        self._save()
    
    def export_anchors(self, filepath: str) -> None:
        """Export anchors to a JSON file."""
        data = {
            "anchors": [a.to_dict() for a in self._anchors],
            "exported_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def import_anchors(self, filepath: str, merge: bool = True) -> int:
        """Import anchors from a JSON file. Returns number of new anchors added."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        imported = [Anchor.from_dict(a) for a in data.get("anchors", [])]
        
        if not merge:
            self._anchors = imported
            self._save()
            return len(imported)
        
        # Merge: add only new anchors
        existing_ids = {a.id for a in self._anchors}
        new_count = 0
        for anchor in imported:
            if anchor.id not in existing_ids:
                self._anchors.append(anchor)
                existing_ids.add(anchor.id)
                new_count += 1
        
        if new_count > 0:
            self._save()
        
        return new_count

