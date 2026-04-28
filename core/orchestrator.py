from __future__ import annotations
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import MAX_QC_RETRIES, OUTPUT_DIR
from .models import PipelineState

console = Console()

STATE_FILE = OUTPUT_DIR / "pipeline_state.json"


class Orchestrator:
    """
    Sequential pipeline orchestrator with a QC feedback loop.

    Flow:
        PromptUnderstanding → StoryGeneration → SceneBreakdown
        → VisualPrompt → AssetGeneration → VoiceoverAudio
        → AnimationComposition → QualityControl
                                      ↓ (fail, retry)
                              AnimationComposition

    State is saved to output/pipeline_state.json after every agent so
    the run can be resumed with --resume if it crashes mid-way.
    """

    def __init__(self) -> None:
        from agents.prompt_understanding_agent import PromptUnderstandingAgent
        from agents.story_generation_agent import StoryGenerationAgent
        from agents.scene_breakdown_agent import SceneBreakdownAgent
        from agents.visual_prompt_agent import VisualPromptAgent
        from agents.asset_generation_agent import AssetGenerationAgent
        from agents.voiceover_audio_agent import VoiceoverAudioAgent
        from agents.animation_composition_agent import AnimationCompositionAgent
        from agents.quality_control_agent import QualityControlAgent

        self.pipeline = [
            PromptUnderstandingAgent(),
            StoryGenerationAgent(),
            SceneBreakdownAgent(),
            VisualPromptAgent(),
            AssetGenerationAgent(),
            VoiceoverAudioAgent(),
            AnimationCompositionAgent(),
        ]
        self.qc_agent = QualityControlAgent()
        self.composition_agent = AnimationCompositionAgent()

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def run(self, user_prompt: str, resume: bool = False) -> PipelineState:
        state = self._load_state(user_prompt) if resume else PipelineState(
            user_prompt=user_prompt, status="running"
        )
        state.status = "running"

        console.print(
            Panel.fit(
                f"[bold cyan]Animation Video Generator[/bold cyan]\n"
                f"Prompt: [italic]{user_prompt}[/italic]"
                + (" [yellow](resuming)[/yellow]" if resume else ""),
                border_style="cyan",
            )
        )

        # ── Main sequential pipeline ──────────────────────────────────── #
        for agent in self.pipeline:
            if agent.name in state.completed_stages:
                console.print(f"[dim]  ↷ Skipping {agent.name} (already done)[/dim]")
                continue
            state = agent.run(state)
            self._save_state(state)
            if state.status == "error":
                console.print(
                    f"[bold red]Pipeline halted at {agent.name}[/bold red]\n"
                    f"[yellow]Resume with:[/yellow] py main.py --resume"
                )
                return state

        # ── QC loop with retries ─────────────────────────────────────── #
        for attempt in range(MAX_QC_RETRIES + 1):
            if "QualityControlAgent" not in state.completed_stages or attempt > 0:
                state = self.qc_agent.run(state)
                self._save_state(state)
            if state.status == "error":
                return state

            if state.qc_report and state.qc_report.approved:
                break

            if attempt < MAX_QC_RETRIES:
                console.print(
                    f"[yellow]QC not approved (score={state.qc_report.overall_score:.1f}). "
                    f"Re-running composition (attempt {attempt + 1}/{MAX_QC_RETRIES})…[/yellow]"
                )
                state.qc_retry_count += 1
                state.completed_stages = [
                    s for s in state.completed_stages
                    if s not in ("AnimationCompositionAgent", "QualityControlAgent")
                ]
                state = self.composition_agent.run(state)
                self._save_state(state)
                if state.status == "error":
                    return state

        state.status = "completed"
        self._save_state(state)
        self._print_summary(state)
        return state

    # ------------------------------------------------------------------ #
    # State persistence                                                     #
    # ------------------------------------------------------------------ #

    def _save_state(self, state: PipelineState) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(
                json.dumps(state.model_dump(), indent=2, default=str)
            )
        except Exception as exc:
            console.print(f"[yellow]  Warning: could not save state: {exc}[/yellow]")

    def _load_state(self, user_prompt: str) -> PipelineState:
        if not STATE_FILE.exists():
            console.print(
                "[yellow]No saved state found — starting fresh.[/yellow]"
            )
            return PipelineState(user_prompt=user_prompt, status="running")
        try:
            data = json.loads(STATE_FILE.read_text())
            state = PipelineState(**data)
            done = ", ".join(state.completed_stages) or "none"
            console.print(
                f"[green]Resuming from saved state.[/green]\n"
                f"  Completed stages: [cyan]{done}[/cyan]"
            )
            return state
        except Exception as exc:
            console.print(
                f"[yellow]Could not load saved state ({exc}) — starting fresh.[/yellow]"
            )
            return PipelineState(user_prompt=user_prompt, status="running")

    # ------------------------------------------------------------------ #
    # Summary                                                               #
    # ------------------------------------------------------------------ #

    def _print_summary(self, state: PipelineState) -> None:
        table = Table(title="Pipeline Summary", border_style="green")
        table.add_column("Stage", style="bold")
        table.add_column("Status")

        for name in [
            "PromptUnderstandingAgent", "StoryGenerationAgent",
            "SceneBreakdownAgent", "VisualPromptAgent",
            "AssetGenerationAgent", "VoiceoverAudioAgent",
            "AnimationCompositionAgent", "QualityControlAgent",
        ]:
            done = name in state.completed_stages
            table.add_row(name, "[green]✓[/green]" if done else "[red]✗[/red]")

        console.print(table)

        if state.qc_report:
            console.print(
                Panel(
                    f"Overall QC Score: [bold]{state.qc_report.overall_score:.1f}/10[/bold]\n"
                    f"Approved: {'[green]YES[/green]' if state.qc_report.approved else '[red]NO[/red]'}\n"
                    f"Summary: {state.qc_report.improvement_summary}",
                    title="QC Report",
                    border_style="magenta",
                )
            )

        if state.composition_plan:
            console.print(
                Panel(
                    f"Title: [bold]{state.composition_plan.title}[/bold]\n"
                    f"Duration: {state.composition_plan.total_duration_seconds}s\n"
                    f"Resolution: {state.composition_plan.resolution}\n"
                    f"Format: {state.composition_plan.export_format}\n"
                    f"Render: [italic]{state.composition_plan.final_render_command}[/italic]",
                    title="Final Video Plan",
                    border_style="cyan",
                )
            )

        if state.output_video_path:
            console.print(
                Panel(
                    f"[bold green]{state.output_video_path}[/bold green]",
                    title="Final Video",
                    border_style="green",
                )
            )
