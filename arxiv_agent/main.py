"""Main entry point for the arxiv agent."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from .anchors import AnchorStore
from .cli import run_cli
from .feed import fetch_all_feeds
from .matcher import SemanticMatcher
from .models import Config

console = Console()


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch and display papers from feeds."""
    config = Config()
    
    if args.feeds:
        config.feeds = args.feeds
    
    if args.threshold:
        config.relevance_threshold = args.threshold
    
    if args.max_results:
        config.max_results = args.max_results
    
    store = AnchorStore(config)
    matcher = SemanticMatcher(config)
    
    console.print("[dim]Fetching papers from arxiv...[/dim]")
    papers = fetch_all_feeds(config.feeds)
    console.print(f"[dim]Found {len(papers)} papers[/dim]")
    
    anchors = store.anchors
    if anchors:
        console.print("[dim]Filtering by relevance...[/dim]")
        papers = matcher.filter_papers(papers, anchors)
    
    # Output as JSON for piping
    if args.json:
        output = [p.to_dict() for p in papers]
        print(json.dumps(output, indent=2))
    else:
        for p in papers:
            score_str = f"({p.relevance_score:.0%})" if p.relevance_score > 0 else ""
            console.print(f"[bold]{p.title}[/bold] {score_str}")
            console.print(f"  [dim]{p.link}[/dim]")
            console.print()


def cmd_topics(args: argparse.Namespace) -> None:
    """Manage topics."""
    config = Config()
    store = AnchorStore(config)
    
    if args.add:
        anchor = store.add_topic(args.add, args.title)
        console.print(f"[green]Added topic: {anchor.title}[/green]")
    
    elif args.remove:
        if store.remove_anchor(args.remove):
            console.print(f"[green]Removed anchor: {args.remove}[/green]")
        else:
            console.print(f"[red]Anchor not found: {args.remove}[/red]")
            sys.exit(1)
    
    elif args.list:
        topics = store.get_topics()
        if not topics:
            console.print("[yellow]No topics defined[/yellow]")
        else:
            for t in topics:
                console.print(f"[bold]{t.id}[/bold]: {t.title}")
    
    elif args.clear:
        store.clear_all()
        console.print("[green]Cleared all anchors[/green]")


def cmd_export(args: argparse.Namespace) -> None:
    """Export anchors to a file."""
    config = Config()
    store = AnchorStore(config)
    store.export_anchors(args.output)
    console.print(f"[green]Exported {len(store.anchors)} anchors to {args.output}[/green]")


def cmd_import(args: argparse.Namespace) -> None:
    """Import anchors from a file."""
    config = Config()
    store = AnchorStore(config)
    count = store.import_anchors(args.input, merge=not args.replace)
    console.print(f"[green]Imported {count} new anchors from {args.input}[/green]")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="arxiv Agent - Your personal research paper curator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arxiv-agent                     # Run interactive mode
  arxiv-agent web                 # Start web interface
  arxiv-agent fetch               # Fetch and filter papers
  arxiv-agent fetch --json        # Output as JSON
  arxiv-agent topics --add "Large language model training"
  arxiv-agent topics --list
  arxiv-agent export anchors.json
  arxiv-agent import anchors.json
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # web command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    web_parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    
    # fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch and filter papers")
    fetch_parser.add_argument("--feeds", nargs="+", help="RSS feed URLs")
    fetch_parser.add_argument("--threshold", type=float, help="Relevance threshold (0.0-1.0)")
    fetch_parser.add_argument("--max-results", type=int, help="Maximum number of results")
    fetch_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # topics command
    topics_parser = subparsers.add_parser("topics", help="Manage topics")
    topics_group = topics_parser.add_mutually_exclusive_group()
    topics_group.add_argument("--add", metavar="TOPIC", help="Add a new topic")
    topics_group.add_argument("--remove", metavar="ID", help="Remove an anchor by ID")
    topics_group.add_argument("--list", action="store_true", help="List all topics")
    topics_group.add_argument("--clear", action="store_true", help="Clear all anchors")
    topics_parser.add_argument("--title", help="Short title for the topic")
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export anchors")
    export_parser.add_argument("output", help="Output file path")
    
    # import command
    import_parser = subparsers.add_parser("import", help="Import anchors")
    import_parser.add_argument("input", help="Input file path")
    import_parser.add_argument("--replace", action="store_true", help="Replace existing anchors")
    
    args = parser.parse_args()
    
    if args.command == "web":
        from .web import run_server
        run_server(host=args.host, port=args.port)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "topics":
        cmd_topics(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    else:
        # Default: run interactive mode
        run_cli()


if __name__ == "__main__":
    main()

