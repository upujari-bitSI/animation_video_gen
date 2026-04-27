"""
Animation Video Generator — Multi-Agent Pipeline (local models)
===============================================================
Usage:
    python main.py "A small star learns to shine despite being afraid"
    python main.py --prompt "A brave kitten saves the forest" --output result.json
    python main.py --interactive
    python main.py --example
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from config import OLLAMA_HOST, OLLAMA_MODEL
from core.orchestrator import Orchestrator

console = Console()

EXAMPLE_PROMPT = "A small star learns to shine despite being afraid"


def run(prompt: str, output_path: str | None = None) -> dict:
    console.print(
        f"[dim]LLM : {OLLAMA_MODEL} @ {OLLAMA_HOST}[/dim]\n"
        "[dim]Make sure Ollama is running: [bold]ollama serve[/bold][/dim]"
    )

    orchestrator = Orchestrator()
    state = orchestrator.run(prompt)

    if state.output_video_path:
        console.print(
            f"\n[bold green]Final video:[/bold green] {state.output_video_path}"
        )

    result = state.model_dump()

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        console.print(f"\n[green]Full pipeline state saved to:[/green] {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Animation Video Generator"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Animation prompt (positional)",
    )
    parser.add_argument(
        "--prompt", "-p",
        dest="prompt_flag",
        help="Animation prompt (flag form)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save the full pipeline JSON output",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter prompt interactively",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help=f'Run with the built-in example prompt: "{EXAMPLE_PROMPT}"',
    )

    args = parser.parse_args()

    if args.example:
        prompt = EXAMPLE_PROMPT
    elif args.interactive:
        prompt = Prompt.ask("[bold cyan]Enter your animation prompt[/bold cyan]")
    elif args.prompt_flag:
        prompt = args.prompt_flag
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        console.print(
            f'\n[yellow]Tip:[/yellow] Try: python main.py "{EXAMPLE_PROMPT}"'
        )
        sys.exit(0)

    run(prompt.strip(), output_path=args.output)


if __name__ == "__main__":
    main()
