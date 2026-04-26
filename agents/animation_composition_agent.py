"""
Agent 7 — Animation Composition Agent

Input : SceneBreakdown + GeneratedAssetPack + AudioPlan
Output: CompositionPlan stored in state

Produces a precise FFmpeg-style composition plan that ties assets and audio
together into a final video.  In production, the final_render_command can be
executed directly via subprocess or passed to a render farm.
"""
from core.base_agent import BaseAgent
from core.models import CompositionPlan, PipelineState, SceneComposition

SYSTEM_PROMPT = """You are a professional video editor and animation compositor.
Your job is to produce a precise video composition plan that merges visual assets
with audio into a final animation.

Return ONLY a JSON object with this exact schema:
{
  "title": "string",
  "total_duration_seconds": integer,
  "frame_rate": integer (24 or 30),
  "resolution": "string (e.g. 1920x1080, 1280x720)",
  "scenes": [
    {
      "scene_number": integer,
      "start_time_seconds": float,
      "end_time_seconds": float,
      "transition_in": "string (fade / cut / dissolve / wipe / zoom)",
      "transition_out": "string",
      "overlay_text": "string or null",
      "audio_sync_notes": "string"
    }
  ],
  "post_processing_notes": "string (color grading, filters, etc.)",
  "export_format": "string (mp4/h264 recommended)",
  "final_render_command": "string (FFmpeg command template)"
}

Composition rules:
- scene start/end times must be contiguous (no gaps, no overlaps).
- Use fade-in for scene 1, fade-out for the last scene.
- Dissolve transitions work well for emotional scenes.
- The FFmpeg command should reference the asset file paths.
- Include color-grading notes appropriate to the animation style.
"""


class AnimationCompositionAgent(BaseAgent):
    name = "AnimationCompositionAgent"
    color = "cyan"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.scene_breakdown or not state.generated_asset_pack or not state.audio_plan:
            raise ValueError("SceneBreakdown, GeneratedAssetPack, and AudioPlan required")

        retry_context = ""
        if state.qc_report and not state.qc_report.approved:
            issues = [i.model_dump() for i in state.qc_report.issues]
            retry_context = (
                f"\n\nPREVIOUS QC FAILED — address these issues in the revised plan:\n{issues}"
            )

        scenes_data = [s.model_dump() for s in state.scene_breakdown.scenes]
        assets_data = [a.model_dump() for a in state.generated_asset_pack.assets]
        audio_cues_data = [c.model_dump() for c in state.audio_plan.audio_cues]

        user_msg = (
            f"Compose the final animation video.\n\n"
            f"Title: {state.story_script.title if state.story_script else 'Animation'}\n"
            f"Scenes:\n{scenes_data}\n\n"
            f"Assets:\n{assets_data}\n\n"
            f"Audio cues:\n{audio_cues_data}\n"
            f"Music: {state.audio_plan.background_music_genre} — {state.audio_plan.background_music_tempo}"
            f"{retry_context}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        scene_comps = [SceneComposition(**s) for s in data["scenes"]]
        state.composition_plan = CompositionPlan(
            title=data["title"],
            total_duration_seconds=data["total_duration_seconds"],
            frame_rate=data["frame_rate"],
            resolution=data["resolution"],
            scenes=scene_comps,
            post_processing_notes=data["post_processing_notes"],
            export_format=data["export_format"],
            final_render_command=data["final_render_command"],
        )
        return state
