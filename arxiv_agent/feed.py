"""RSS feed parser for arxiv papers."""

import random
import re
import time
from datetime import datetime

import feedparser
import httpx

from .models import Paper

USER_AGENT = "arxiv_agent/0.1 (+https://github.com/dawenl/arxiv_agent)"

# arXiv asks for ≥3s between API requests; apply the same to RSS to be polite.
INTER_REQUEST_DELAY = 3.0
MAX_RETRIES = 3
BACKOFF_BASE = 2.0
BACKOFF_CAP = 30.0
RETRY_STATUSES = {429, 500, 502, 503, 504}


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a Retry-After header in delta-seconds form. Returns None for HTTP-date."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    return min(BACKOFF_CAP, BACKOFF_BASE ** attempt) + random.uniform(0, 1)


def parse_arxiv_id(link: str) -> str:
    """Extract arxiv ID from a link."""
    # Handle both http://arxiv.org/abs/2401.12345 and arxiv:2401.12345
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", link)
    if match:
        return match.group(1)
    return link.split("/")[-1]


def parse_authors(entry: dict) -> list[str]:
    """Parse author list from feed entry."""
    if "authors" in entry:
        return [a.get("name", str(a)) for a in entry.get("authors", [])]
    if "author" in entry:
        author = entry["author"]
        if isinstance(author, str):
            return [a.strip() for a in author.split(",")]
        return [author]
    return []


def parse_categories(entry: dict) -> list[str]:
    """Parse categories from feed entry."""
    categories = []
    for tag in entry.get("tags", []):
        if "term" in tag:
            categories.append(tag["term"])
    return categories


def clean_abstract(summary: str) -> str:
    """Clean up the abstract text."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", summary)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_new_or_cross(entry: dict) -> bool:
    """
    Check if paper is a new submission or cross-listing (not a replacement).

    arXiv RSS feeds include an 'arxiv_announce_type' field:
    - 'new' - New submission
    - 'cross' - Cross-listed from another category
    - 'replace' - Replacement/update of existing paper
    - 'replace-cross' - Replacement that's cross-listed
    """
    announce_type = entry.get("arxiv_announce_type", "new").lower()
    return announce_type in ("new", "cross")


def parse_date(date_str: str | None) -> datetime:
    """Parse a date string into datetime."""
    if not date_str:
        return datetime.now()

    try:
        # Try parsing common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return datetime.now()
    except Exception:
        return datetime.now()


def fetch_feed(url: str, timeout: float = 30.0) -> list[Paper]:
    """Fetch and parse an arxiv RSS feed, filtering to only new/cross papers.

    Retries on transient HTTP errors (429, 5xx) with exponential backoff,
    honoring Retry-After when present.
    """
    content: str | None = None
    with httpx.Client(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.get(url)
            except httpx.HTTPError as e:
                if attempt >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to fetch feed {url}: {type(e).__name__}: {e}"
                    ) from e
                time.sleep(_backoff(attempt))
                continue

            if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                wait = _retry_after_seconds(response) or _backoff(attempt)
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Failed to fetch feed {url}: HTTP {response.status_code}"
                )

            content = response.text
            break

    assert content is not None  # loop always breaks or raises

    feed = feedparser.parse(content)
    papers = []

    for entry in feed.entries:
        # Skip replacement papers - only keep new and cross-listed
        if not is_new_or_cross(entry):
            continue

        title = entry.get("title", "No title")
        arxiv_id = parse_arxiv_id(entry.get("id", entry.get("link", "")))

        # Get PDF link
        pdf_link = None
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_link = link.get("href")
                break

        if not pdf_link:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        paper = Paper(
            id=arxiv_id,
            title=title,
            abstract=clean_abstract(entry.get("summary", "")),
            authors=parse_authors(entry),
            categories=parse_categories(entry),
            published=parse_date(entry.get("published")),
            updated=parse_date(entry.get("updated", entry.get("published"))),
            link=entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
            pdf_link=pdf_link,
        )
        papers.append(paper)

    return papers


def fetch_all_feeds(urls: list[str]) -> list[Paper]:
    """Fetch papers from multiple RSS feeds, deduplicating by ID."""
    seen_ids: set[str] = set()
    all_papers: list[Paper] = []

    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(INTER_REQUEST_DELAY)
        try:
            papers = fetch_feed(url)
            for paper in papers:
                if paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    all_papers.append(paper)
        except Exception as e:
            # Log but continue with other feeds
            print(f"Warning: Failed to fetch {url}: {e}")

    # Sort by updated date, newest first
    all_papers.sort(key=lambda p: p.updated, reverse=True)
    return all_papers
