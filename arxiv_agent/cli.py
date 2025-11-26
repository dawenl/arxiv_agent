"""Interactive CLI interface for the arxiv agent."""

import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .anchors import AnchorStore
from .feed import fetch_all_feeds
from .matcher import SemanticMatcher
from .models import Config, Paper

# Custom theme for a distinctive look
CUSTOM_THEME = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "title": "bold #ff6b6b",
    "score.high": "bold #2ecc71",
    "score.medium": "bold #f1c40f",
    "score.low": "dim #e74c3c",
    "category": "italic #9b59b6",
    "author": "dim #3498db",
    "prompt": "bold #e91e63",
})

console = Console(theme=CUSTOM_THEME)


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header() -> None:
    """Print the app header."""
    header = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   █████╗ ██████╗ ██╗  ██╗██╗██╗   ██╗     █████╗  ██████╗ ███████╗██╗   ██╗████████╗ ║
║  ██╔══██╗██╔══██╗╚██╗██╔╝██║██║   ██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝ ║
║  ███████║██████╔╝ ╚███╔╝ ██║██║   ██║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║    ║
║  ██╔══██║██╔══██╗ ██╔██╗ ██║╚██╗ ██╔╝    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║    ║
║  ██║  ██║██║  ██║██╔╝ ██╗██║ ╚████╔╝     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║    ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ║
║                                                                   ║
║              Your Personal Research Paper Curator                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    console.print(header, style="bold #ff6b6b")


def print_paper(paper: Paper, index: int, show_abstract: bool = False) -> None:
    """Print a single paper entry."""
    # Determine score style
    if paper.relevance_score >= 0.6:
        score_style = "score.high"
    elif paper.relevance_score >= 0.45:
        score_style = "score.medium"
    else:
        score_style = "score.low"
    
    # Build the title line
    title_line = Text()
    title_line.append(f"[{index}] ", style="bold #ff6b6b")
    title_line.append(paper.title, style="bold white")
    title_line.append(f"  ({paper.relevance_score:.0%})", style=score_style)
    
    console.print(title_line)
    
    # Authors
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += f" + {len(paper.authors) - 3} more"
    console.print(f"   ✍️  {authors_str}", style="author")
    
    # Categories
    cats_str = " • ".join(paper.categories[:3])
    console.print(f"   🏷️  {cats_str}", style="category")
    
    # Links
    console.print(f"   🔗 {paper.link}", style="dim")
    
    if show_abstract:
        console.print()
        abstract_panel = Panel(
            paper.abstract,
            title="Abstract",
            border_style="dim #9b59b6",
            padding=(1, 2),
        )
        console.print(abstract_panel)
    
    console.print()


def display_papers(papers: list[Paper]) -> None:
    """Display the filtered papers."""
    if not papers:
        console.print("\n[warning]No papers matched your interests.[/warning]")
        console.print("Try adding more topics or adjusting the relevance threshold.\n")
        return
    
    console.print(f"\n[success]Found {len(papers)} relevant papers![/success]\n")
    
    for i, paper in enumerate(papers, 1):
        print_paper(paper, i)


def display_anchors(store: AnchorStore) -> None:
    """Display current anchors."""
    topics = store.get_topics()
    papers = store.get_papers()
    
    if not topics and not papers:
        console.print("\n[warning]No anchors defined yet.[/warning]")
        console.print("Add topics or save papers to start filtering!\n")
        return
    
    if topics:
        console.print("\n[title]📌 Topics[/title]")
        topic_table = Table(show_header=True, header_style="bold #ff6b6b")
        topic_table.add_column("ID", style="dim", width=10)
        topic_table.add_column("Topic", style="white")
        topic_table.add_column("Added", style="dim")
        
        for anchor in topics:
            topic_table.add_row(
                anchor.id,
                anchor.title,
                anchor.added_at.strftime("%Y-%m-%d"),
            )
        console.print(topic_table)
    
    if papers:
        console.print("\n[title]📄 Saved Papers[/title]")
        paper_table = Table(show_header=True, header_style="bold #ff6b6b")
        paper_table.add_column("ID", style="dim", width=14)
        paper_table.add_column("Title", style="white", max_width=60)
        paper_table.add_column("Added", style="dim")
        
        for anchor in papers:
            paper_table.add_row(
                anchor.id,
                anchor.title[:60] + ("..." if len(anchor.title) > 60 else ""),
                anchor.added_at.strftime("%Y-%m-%d"),
            )
        console.print(paper_table)
    
    console.print()


def add_topic_interactive(store: AnchorStore) -> None:
    """Interactively add a new topic."""
    console.print("\n[title]Add a New Topic[/title]")
    console.print("Describe your research interest in a few sentences.")
    console.print("Example: 'Large language model alignment techniques including RLHF and constitutional AI'\n")
    
    topic = Prompt.ask("[prompt]Topic description[/prompt]")
    if not topic.strip():
        console.print("[warning]No topic entered.[/warning]")
        return
    
    title = Prompt.ask("[prompt]Short title (optional)[/prompt]", default=topic[:50])
    
    anchor = store.add_topic(topic, title)
    console.print(f"\n[success]✓ Added topic: {anchor.title}[/success]\n")


def remove_anchor_interactive(store: AnchorStore) -> None:
    """Interactively remove an anchor."""
    display_anchors(store)
    
    if not store.anchors:
        return
    
    anchor_id = Prompt.ask("[prompt]Enter anchor ID to remove (or 'cancel')[/prompt]")
    
    if anchor_id.lower() == "cancel":
        return
    
    if store.remove_anchor(anchor_id):
        console.print(f"\n[success]✓ Removed anchor: {anchor_id}[/success]\n")
    else:
        console.print(f"\n[danger]Anchor not found: {anchor_id}[/danger]\n")


def save_paper_interactive(papers: list[Paper], store: AnchorStore) -> None:
    """Interactively save a paper as an anchor."""
    if not papers:
        console.print("\n[warning]No papers to save. Fetch papers first.[/warning]\n")
        return
    
    console.print("\n[title]Save Paper to Anchors[/title]")
    
    try:
        idx = IntPrompt.ask(
            "[prompt]Enter paper number (0 to cancel)[/prompt]",
            default=0,
        )
    except KeyboardInterrupt:
        return
    
    if idx == 0:
        return
    
    if idx < 1 or idx > len(papers):
        console.print(f"\n[danger]Invalid paper number. Choose 1-{len(papers)}[/danger]\n")
        return
    
    paper = papers[idx - 1]
    anchor = store.add_paper(paper)
    console.print(f"\n[success]✓ Saved paper: {paper.title[:60]}...[/success]\n")


def view_paper_details(papers: list[Paper]) -> None:
    """View detailed information about a paper."""
    if not papers:
        console.print("\n[warning]No papers available.[/warning]\n")
        return
    
    try:
        idx = IntPrompt.ask(
            "[prompt]Enter paper number to view details (0 to cancel)[/prompt]",
            default=0,
        )
    except KeyboardInterrupt:
        return
    
    if idx == 0:
        return
    
    if idx < 1 or idx > len(papers):
        console.print(f"\n[danger]Invalid paper number. Choose 1-{len(papers)}[/danger]\n")
        return
    
    paper = papers[idx - 1]
    console.print()
    print_paper(paper, idx, show_abstract=True)


def main_menu(
    store: AnchorStore,
    matcher: SemanticMatcher,
    config: Config,
) -> None:
    """Main interactive loop."""
    current_papers: list[Paper] = []
    
    while True:
        console.print("\n[title]═══ Main Menu ═══[/title]")
        console.print("  [bold]1[/bold] • Fetch & filter today's papers")
        console.print("  [bold]2[/bold] • View current anchors")
        console.print("  [bold]3[/bold] • Add a topic")
        console.print("  [bold]4[/bold] • Save a paper to anchors")
        console.print("  [bold]5[/bold] • View paper details")
        console.print("  [bold]6[/bold] • Remove an anchor")
        console.print("  [bold]7[/bold] • Configure feeds")
        console.print("  [bold]8[/bold] • Adjust threshold")
        console.print("  [bold]q[/bold] • Quit")
        console.print()
        
        choice = Prompt.ask("[prompt]Select option[/prompt]", default="1")
        
        if choice == "1":
            # Fetch and filter papers
            console.print("\n[info]Fetching papers from arxiv feeds...[/info]")
            
            try:
                all_papers = fetch_all_feeds(config.feeds)
                console.print(f"[info]Fetched {len(all_papers)} papers[/info]")
                
                anchors = store.anchors
                if not anchors:
                    console.print("\n[warning]No anchors defined! Add topics first.[/warning]")
                    current_papers = all_papers[:20]  # Show first 20 unfiltered
                else:
                    console.print("[info]Computing relevance scores...[/info]")
                    current_papers = matcher.filter_papers(all_papers, anchors)
                
                display_papers(current_papers)
                
            except Exception as e:
                console.print(f"\n[danger]Error fetching papers: {e}[/danger]\n")
        
        elif choice == "2":
            display_anchors(store)
        
        elif choice == "3":
            add_topic_interactive(store)
        
        elif choice == "4":
            save_paper_interactive(current_papers, store)
        
        elif choice == "5":
            view_paper_details(current_papers)
        
        elif choice == "6":
            remove_anchor_interactive(store)
        
        elif choice == "7":
            console.print("\n[title]Current Feeds[/title]")
            for i, feed in enumerate(config.feeds, 1):
                console.print(f"  {i}. {feed}")
            console.print("\n[info]Edit ~/.arxiv_agent/config.json to modify feeds[/info]\n")
        
        elif choice == "8":
            console.print(f"\n[info]Current threshold: {config.relevance_threshold:.0%}[/info]")
            try:
                new_threshold = Prompt.ask(
                    "[prompt]New threshold (0.0-1.0, or 'cancel')[/prompt]",
                    default=str(config.relevance_threshold),
                )
                if new_threshold.lower() != "cancel":
                    config.relevance_threshold = float(new_threshold)
                    console.print(f"[success]Threshold set to {config.relevance_threshold:.0%}[/success]\n")
            except ValueError:
                console.print("[danger]Invalid threshold value[/danger]\n")
        
        elif choice.lower() == "q":
            console.print("\n[success]👋 Happy researching![/success]\n")
            break


def run_cli() -> None:
    """Run the interactive CLI."""
    clear_screen()
    print_header()
    
    config = Config()
    store = AnchorStore(config)
    
    # Only initialize matcher when needed (lazy loading)
    matcher = SemanticMatcher(config)
    
    # Check if first run
    if not store.anchors:
        console.print("\n[info]Welcome to arxiv Agent! Let's set up your interests.[/info]")
        console.print("[info]First, add some topics that describe your research interests.[/info]\n")
        
        while True:
            add_topic_interactive(store)
            if not Confirm.ask("[prompt]Add another topic?[/prompt]"):
                break
    
    main_menu(store, matcher, config)

