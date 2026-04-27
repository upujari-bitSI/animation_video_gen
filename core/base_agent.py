from __future__ import annotations
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from rich.console import Console

from config import OLLAMA_HOST, OLLAMA_MODEL
from .models import PipelineState

console = Console()


class BaseAgent(ABC):
    """Base class for all pipeline agents. Uses Ollama (local LLM) for reasoning."""

    name: str = "BaseAgent"
    color: str = "white"

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
    # LLM call via Ollama                                                   #
    # ------------------------------------------------------------------ #

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        retries: int = 3,
    ) -> str:
        """Call a local Ollama model and return the response text."""
        import ollama

        client = ollama.Client(host=OLLAMA_HOST)

        for attempt in range(retries):
            try:
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    format="json",
                    options={"temperature": 0.7, "num_ctx": 8192},
                )
                return response.message.content
            except Exception as exc:
                err = str(exc)
                if "connection" in err.lower() or "refused" in err.lower():
                    raise RuntimeError(
                        f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
                        "  → Start it with: ollama serve\n"
                        f"  → Then pull the model: ollama pull {OLLAMA_MODEL}"
                    ) from exc
                if attempt == retries - 1:
                    raise RuntimeError(f"Ollama call failed after {retries} tries: {exc}") from exc
                wait = 2 ** attempt
                console.print(f"[yellow]  Retry {attempt+1}/{retries} in {wait}s…[/yellow]")
                time.sleep(wait)

        raise RuntimeError("LLM call failed")  # unreachable, satisfies type checker

    # ------------------------------------------------------------------ #
    # JSON parsing helper                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract the first JSON object/array from a text response."""
        # Markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            return json.loads(match.group(1))
        # Bare JSON object or array
        match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
        if match:
            return json.loads(match.group(1))
        # Whole text is JSON (Ollama format=json path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"No JSON found in LLM response.\n"
                f"First 300 chars: {text[:300]}"
            )
