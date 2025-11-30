"""Web interface for the arxiv agent."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .anchors import AnchorStore
from .feed import fetch_all_feeds
from .matcher import SemanticMatcher
from .models import ARXIV_CATEGORIES, EMBEDDING_MODELS, Config, Paper

# Initialize app
app = FastAPI(title="arxiv Agent", description="Your personal research paper curator")

# Global state
_config: Config | None = None
_store: AnchorStore | None = None
_matcher: SemanticMatcher | None = None
_cached_papers: list[Paper] = []


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_store() -> AnchorStore:
    global _store
    if _store is None:
        _store = AnchorStore(get_config())
    return _store


def get_matcher() -> SemanticMatcher:
    global _matcher
    if _matcher is None:
        _matcher = SemanticMatcher(get_config())
    return _matcher


# Request/Response models
class TopicCreate(BaseModel):
    text: str
    title: str | None = None


class TopicResponse(BaseModel):
    id: str
    type: str
    title: str
    text: str
    added_at: str


class PaperResponse(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    link: str
    pdf_link: str | None
    relevance_score: float


class FetchParams(BaseModel):
    threshold: float | None = None
    max_results: int | None = None


class PapersResult(BaseModel):
    papers: list[PaperResponse]
    total_in_feed: int  # Total papers fetched from RSS before filtering


class SettingsUpdate(BaseModel):
    threshold: float | None = None
    max_results: int | None = None
    categories: list[str] | None = None
    embedding_model: str | None = None


class CategoryInfo(BaseModel):
    id: str
    name: str
    selected: bool


class EmbeddingModelInfo(BaseModel):
    id: str
    name: str
    selected: bool


# API Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/anchors")
async def list_anchors() -> list[TopicResponse]:
    """List all anchors (topics and papers)."""
    store = get_store()
    return [
        TopicResponse(
            id=a.id,
            type=a.type,
            title=a.title,
            text=a.text,
            added_at=a.added_at.isoformat(),
        )
        for a in store.anchors
    ]


@app.post("/api/anchors/topic")
async def add_topic(topic: TopicCreate) -> TopicResponse:
    """Add a new topic anchor."""
    store = get_store()
    anchor = store.add_topic(topic.text, topic.title)
    return TopicResponse(
        id=anchor.id,
        type=anchor.type,
        title=anchor.title,
        text=anchor.text,
        added_at=anchor.added_at.isoformat(),
    )


@app.post("/api/anchors/paper/{paper_id}")
async def save_paper(paper_id: str) -> TopicResponse:
    """Save a paper as an anchor."""
    global _cached_papers
    
    paper = next((p for p in _cached_papers if p.id == paper_id), None)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found in current results")
    
    store = get_store()
    anchor = store.add_paper(paper)
    return TopicResponse(
        id=anchor.id,
        type=anchor.type,
        title=anchor.title,
        text=anchor.text,
        added_at=anchor.added_at.isoformat(),
    )


@app.delete("/api/anchors/{anchor_id}")
async def remove_anchor(anchor_id: str):
    """Remove an anchor."""
    store = get_store()
    if not store.remove_anchor(anchor_id):
        raise HTTPException(status_code=404, detail="Anchor not found")
    return {"status": "ok"}


@app.get("/api/papers")
async def fetch_papers(
    threshold: float | None = None,
    max_results: int | None = None,
    categories: str | None = None,  # Comma-separated list
) -> PapersResult:
    """Fetch and filter papers from arxiv feeds."""
    global _cached_papers
    
    config = get_config()
    store = get_store()
    matcher = get_matcher()
    
    # Use provided categories or default
    if categories:
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        feeds = [f"https://rss.arxiv.org/rss/{cat}" for cat in cat_list]
    else:
        feeds = config.feeds
    
    # Fetch all papers
    all_papers = fetch_all_feeds(feeds)
    total_in_feed = len(all_papers)
    
    # Filter by relevance if we have anchors
    anchors = store.anchors
    if anchors:
        filtered = matcher.filter_papers(
            all_papers,
            anchors,
            threshold=threshold or config.relevance_threshold,
            max_results=max_results or config.max_results,
        )
    else:
        # No anchors - return all papers (limited)
        filtered = all_papers[: (max_results or config.max_results)]
    
    _cached_papers = filtered
    
    return PapersResult(
        papers=[
            PaperResponse(
                id=p.id,
                title=p.title,
                abstract=p.abstract,
                authors=p.authors,
                categories=p.categories,
                published=p.published.isoformat(),
                updated=p.updated.isoformat(),
                link=p.link,
                pdf_link=p.pdf_link,
                relevance_score=p.relevance_score,
            )
            for p in filtered
        ],
        total_in_feed=total_in_feed,
    )


@app.get("/api/categories")
async def list_categories() -> list[CategoryInfo]:
    """List all available arxiv categories."""
    config = get_config()
    return [
        CategoryInfo(
            id=cat_id,
            name=cat_name,
            selected=cat_id in config.categories,
        )
        for cat_id, cat_name in ARXIV_CATEGORIES.items()
    ]


@app.get("/api/embedding-models")
async def list_embedding_models() -> list[EmbeddingModelInfo]:
    """List all available embedding models."""
    config = get_config()
    return [
        EmbeddingModelInfo(
            id=model_id,
            name=model_name,
            selected=model_id == config.embedding_model,
        )
        for model_id, model_name in EMBEDDING_MODELS.items()
    ]


@app.get("/api/settings")
async def get_settings():
    """Get current settings."""
    config = get_config()
    return {
        "threshold": config.relevance_threshold,
        "max_results": config.max_results,
        "categories": config.categories,
        "embedding_model": config.embedding_model,
    }


@app.put("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update settings."""
    global _matcher
    config = get_config()
    
    if settings.threshold is not None:
        config.relevance_threshold = settings.threshold
    if settings.max_results is not None:
        config.max_results = settings.max_results
    if settings.categories is not None:
        config.categories = settings.categories
    if settings.embedding_model is not None and settings.embedding_model != config.embedding_model:
        # Validate the model
        if settings.embedding_model not in EMBEDDING_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid embedding model: {settings.embedding_model}")
        config.embedding_model = settings.embedding_model
        # Reset matcher to force model reload
        _matcher = None
    
    return {"status": "ok"}


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the web server."""
    import uvicorn
    
    print(f"\n🚀 arxiv Agent running at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

