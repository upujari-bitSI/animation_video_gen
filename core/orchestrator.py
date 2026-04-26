from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import MAX_QC_RETRIES
from .models import PipelineState

console = Console()


class Orchestrator:
    """
    Sequential pipeline orchestrator with a QC feedback loop.

    Flow:
        PromptUnderstanding → StoryGeneration → SceneBreakdown
        → VisualPrompt → AssetGeneration → VoiceoverAudio
        → AnimationComposition → QualityControl
                                      ↓ (fail, retry)
                              AnimationComposition
    """

    def __init__(self) -> None:
        # Import here to avoid circular dependencies at module load time.
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

    def run(self, user_prompt: str) -> PipelineState:
        state = PipelineState(user_prompt=user_prompt, status="running")

        console.print(
            Panel.fit(
                f"[bold cyan]Animation Video Generator[/bold cyan]\n"
                f"Prompt: [italic]{user_prompt}[/italic]",
                border_style="cyan",
            )
        )

        # ── Main sequential pipeline ──────────────────────────────────── #
        for agent in self.pipeline:
            state = agent.run(state)
            if state.status == "error":
                console.print(f"[bold red]Pipeline halted at {agent.name}[/bold red]")
                return state

        # ── QC loop with retries ─────────────────────────────────────── #
        for attempt in range(MAX_QC_RETRIES + 1):
            state = self.qc_agent.run(state)
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
                state = self.composition_agent.run(state)
                if state.status == "error":
                    return state

        state.status = "completed"
        self._print_summary(state)
        return state

    # ------------------------------------------------------------------ #

    def _print_summary(self, state: PipelineState) -> None:
        table = Table(title="Pipeline Summary", border_style="green")
        table.add_column("Stage", style="bold")
        table.add_column("Status")

        stage_names = [
            "PromptUnderstandingAgent",
            "StoryGenerationAgent",
            "SceneBreakdownAgent",
            "VisualPromptAgent",
            "AssetGenerationAgent",
            "VoiceoverAudioAgent",
            "AnimationCompositionAgent",
            "QualityControlAgent",
        ]
        for name in stage_names:
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
