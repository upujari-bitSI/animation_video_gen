"""
Agent 8 — Quality Control Agent

Input : Full pipeline state (all previous outputs)
Output: QCReport stored in state

Performs a holistic review of the entire production:
  • Scene continuity
  • Audio/visual sync
  • Visual consistency
  • Pacing and timing
  • Story coherence

If the overall_score < 7.0, approved=False triggers a retry loop
in the Orchestrator.
"""
from core.base_agent import BaseAgent
from core.models import PipelineState, QCIssue, QCReport

SYSTEM_PROMPT = """You are a senior quality control supervisor for an animation studio.
Review all production materials and provide a detailed QC report.

Return ONLY a JSON object with this exact schema:
{
  "overall_score": float (0.0-10.0),
  "continuity_score": float (0.0-10.0),
  "audio_sync_score": float (0.0-10.0),
  "visual_consistency_score": float (0.0-10.0),
  "pacing_score": float (0.0-10.0),
  "issues": [
    {
      "severity": "critical | warning | suggestion",
      "category": "continuity | audio_sync | visual | pacing | story",
      "scene_reference": integer or null,
      "description": "string",
      "suggested_fix": "string"
    }
  ],
  "approved": boolean (true if overall_score >= 7.0 and no critical issues),
  "improvement_summary": "string"
}

QC checklist:
1. CONTINUITY: Do characters look/act consistently across scenes?
2. AUDIO SYNC: Does narration timing fit each scene's duration?
3. VISUAL: Are scene transitions logical? Is the color palette consistent?
4. PACING: Does scene duration match the narrative weight?
5. STORY: Does the final composition faithfully represent the script?

Be constructive but strict. Approve only if the animation meets professional standards.
"""


class QualityControlAgent(BaseAgent):
    name = "QualityControlAgent"
    color = "bright_white"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.composition_plan:
            raise ValueError("CompositionPlan required")

        scenes_data = (
            [s.model_dump() for s in state.scene_breakdown.scenes]
            if state.scene_breakdown else []
        )
        vp_data = (
            [p.model_dump() for p in state.visual_prompt_pack.prompts]
            if state.visual_prompt_pack else []
        )
        audio_cues_data = (
            [c.model_dump() for c in state.audio_plan.audio_cues]
            if state.audio_plan else []
        )
        comp_scenes_data = [s.model_dump() for s in state.composition_plan.scenes]

        user_msg = (
            f"Perform QC on this animation production.\n\n"
            f"Story title: {state.story_script.title if state.story_script else 'N/A'}\n"
            f"Emotional arc: {state.story_script.emotional_arc if state.story_script else 'N/A'}\n\n"
            f"Scene breakdown:\n{scenes_data}\n\n"
            f"Visual prompts:\n{vp_data}\n\n"
            f"Audio cues:\n{audio_cues_data}\n\n"
            f"Composition plan:\n{comp_scenes_data}\n"
            f"Total duration: {state.composition_plan.total_duration_seconds}s\n"
            f"Resolution: {state.composition_plan.resolution}\n"
            f"Post-processing: {state.composition_plan.post_processing_notes}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        issues = [QCIssue(**i) for i in data.get("issues", [])]
        state.qc_report = QCReport(
            overall_score=data["overall_score"],
            continuity_score=data["continuity_score"],
            audio_sync_score=data["audio_sync_score"],
            visual_consistency_score=data["visual_consistency_score"],
            pacing_score=data["pacing_score"],
            issues=issues,
            approved=data["approved"],
            improvement_summary=data["improvement_summary"],
        )
        return state
