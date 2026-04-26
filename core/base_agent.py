from __future__ import annotations
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import anthropic
from rich.console import Console

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS
from .models import PipelineState

console = Console()


class BaseAgent(ABC):
    """Base class for all pipeline agents."""

    name: str = "BaseAgent"
    color: str = "white"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = ANTHROPIC_MODEL

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def run(self, state: PipelineState) -> PipelineState:
        console.print(f"\n[bold {self.color}]▶ {self.name}[/bold {self.color}]")
        try:
            state = self._execute(state)
            state.completed_stages.append(self.name)
            console.print(f"[green]✓ {self.name} completed[/green]")
        except Exception as exc:
            state.errors.append(f"{self.name}: {exc}")
            state.status = "error"
            console.print(f"[red]✗ {self.name} failed: {exc}[/red]")
        return state

    # ------------------------------------------------------------------ #
    # Subclass contract                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _execute(self, state: PipelineState) -> PipelineState:
        ...

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = MAX_TOKENS,
        retries: int = 3,
    ) -> str:
        """Call Claude with retry logic and prompt caching on the system prompt."""
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                console.print(f"[yellow]Rate limit hit, retrying in {wait}s…[/yellow]")
                time.sleep(wait)
            except anthropic.APIError as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
                console.print(f"[yellow]API error ({exc}), retrying…[/yellow]")
        raise RuntimeError(f"{self.name}: LLM call failed after {retries} retries")

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract the first JSON object or array from a text response."""
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
        if match:
            return json.loads(match.group(1))
        raise ValueError("No JSON found in LLM response")
