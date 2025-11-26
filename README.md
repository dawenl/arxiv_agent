# arxiv Agent 🔬

Your personal AI-powered research paper curator. Feed it arxiv RSS feeds, tell it your interests, and it will filter through the daily paper deluge to show you only what matters.

## Features

- 📡 **RSS Feed Parsing** - Automatically fetches papers from multiple arxiv categories
- 🧠 **Semantic Matching** - Uses embeddings to understand paper relevance beyond keywords
- 📌 **Interest Anchors** - Define topics or save papers as reference points for matching
- 🌐 **Web Interface** - Beautiful, modern web UI for browsing papers
- 🎨 **Beautiful CLI** - Interactive terminal interface with rich formatting
- 📤 **Import/Export** - Share your interest profiles with colleagues

## Installation

```bash
cd arxiv_agent
uv pip install -e .
```

Or run directly:

```bash
uv run python -m arxiv_agent.main
```

## Quick Start

### Web Interface (Recommended)

```bash
# Start the web server
uv run python -m arxiv_agent.main web

# Or with custom host/port
uv run python -m arxiv_agent.main web --host 0.0.0.0 --port 8080
```

Then open http://127.0.0.1:8765 in your browser.

### Interactive CLI Mode

```bash
arxiv-agent
```

On first run, you'll be prompted to add topics that describe your research interests. Examples:
- "Large language model alignment and RLHF techniques"
- "Efficient transformer architectures and attention mechanisms"
- "Multimodal learning combining vision and language"

### CLI Mode

```bash
# Add topics
arxiv-agent topics --add "Reinforcement learning from human feedback"
arxiv-agent topics --add "Constitutional AI and AI safety"

# List your topics
arxiv-agent topics --list

# Fetch filtered papers
arxiv-agent fetch

# Output as JSON (for scripting)
arxiv-agent fetch --json > papers.json

# Adjust filtering threshold (0.0 - 1.0)
arxiv-agent fetch --threshold 0.5

# Fetch from specific feeds
arxiv-agent fetch --feeds https://rss.arxiv.org/rss/cs.LG https://rss.arxiv.org/rss/cs.AI
```

### Export/Import

```bash
# Export your interests
arxiv-agent export my_interests.json

# Import interests (merges by default)
arxiv-agent import colleague_interests.json

# Replace all interests
arxiv-agent import new_interests.json --replace
```

## How It Works

1. **Anchors**: Your interests are represented as "anchors" - either topics (text descriptions) or saved papers
2. **Embeddings**: Both anchors and new papers are converted to dense vector embeddings using `sentence-transformers`
3. **Similarity**: Papers are ranked by their maximum cosine similarity to any anchor
4. **Filtering**: Papers above the relevance threshold are shown, sorted by relevance

## Default Categories

The agent monitors these arxiv categories by default:
- `cs.LG` - Machine Learning
- `cs.AI` - Artificial Intelligence
- `cs.CL` - Computation and Language
- `cs.IR` - Information Retrieval

You can select from 20+ categories in the web interface including:
- Computer Science: CV, NE, RO, HC, SE, DC, CR, DB, PL
- Statistics: stat.ML, stat.TH
- Math: math.OC, math.ST
- Electrical Engineering: eess.AS, eess.IV, eess.SP
- And more!

You can also filter papers by date using the date picker.

## Data Storage

All data is stored in `~/.arxiv_agent/`:
- `anchors.json` - Your topics and saved papers
- `embeddings_cache.json` - Cached embeddings for faster repeated runs

## Tips

1. **Be specific with topics**: "Contrastive learning for vision-language models" works better than "machine learning"
2. **Save papers**: When you find a great paper, save it as an anchor. Future similar papers will rank higher.
3. **Tune the threshold**: Start with 0.35 (default) and adjust. Lower = more papers, higher = stricter filtering.
4. **Use multiple topics**: Add 3-5 focused topics for best results.

## Requirements

- Python 3.10+
- Internet connection (for fetching feeds and downloading embedding model on first run)

## License

MIT

