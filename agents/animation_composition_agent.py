"""
Agent 7 — Animation Composition Agent

Step 1: Uses the local LLM to produce a detailed composition plan
        (timing, transitions, post-processing notes, FFmpeg command template).
Step 2: Calls MoviePy to actually stitch scenes + audio into a final MP4.

Output video saved to: output/video/<title>.mp4
"""
from pathlib import Path

from config import VIDEO_DIR, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH
from core.base_agent import BaseAgent
from core.models import CompositionPlan, PipelineState, SceneComposition
from utils.video_composer import compose_video

SYSTEM_PROMPT = """You are a professional video editor and animation compositor.
Produce a precise video composition plan that merges visual assets with audio.

Return ONLY a JSON object with this exact schema:
{
  "title": "string",
  "total_duration_seconds": integer,
  "frame_rate": integer (24 or 30),
  "resolution": "string (e.g. 1280x720)",
  "scenes": [
    {
      "scene_number": integer,
      "start_time_seconds": float,
      "end_time_seconds": float,
      "transition_in": "string (fade / cut / dissolve / wipe)",
      "transition_out": "string",
      "overlay_text": "string or null",
      "audio_sync_notes": "string"
    }
  ],
  "post_processing_notes": "string (color grading, filters)",
  "export_format": "mp4/h264",
  "final_render_command": "string (FFmpeg command template referencing asset paths)"
}

Rules:
- scene start/end times must be contiguous with no gaps or overlaps.
- Use fade-in for scene 1, fade-out for the last scene.
- The FFmpeg command should reference assets/scene_<N>.png and audio/scene_<N>_narration.wav.
"""


class AnimationCompositionAgent(BaseAgent):
    name = "AnimationCompositionAgent"
    color = "cyan"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.scene_breakdown or not state.generated_asset_pack or not state.audio_plan:
            raise ValueError("SceneBreakdown, GeneratedAssetPack, and AudioPlan required")

        # ── Step 1: LLM produces the composition plan ────────────────── #
        retry_ctx = ""
        if state.qc_report and not state.qc_report.approved:
            issues = [i.model_dump() for i in state.qc_report.issues]
            retry_ctx = f"\n\nPREVIOUS QC FAILED — address these issues:\n{issues}"

        scenes_data = [s.model_dump() for s in state.scene_breakdown.scenes]
        assets_data = [a.model_dump() for a in state.generated_asset_pack.assets]
        cues_data   = [c.model_dump() for c in state.audio_plan.audio_cues]

        user_msg = (
            f"Compose the final animation.\n\n"
            f"Title: {state.story_script.title if state.story_script else 'Animation'}\n"
            f"Scenes:\n{scenes_data}\n\n"
            f"Assets:\n{assets_data}\n\n"
            f"Audio cues:\n{cues_data}\n"
            f"Music: {state.audio_plan.background_music_genre} — "
            f"{state.audio_plan.background_music_tempo}"
            f"{retry_ctx}"
        )

        raw  = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
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

        # ── Step 2: MoviePy renders the actual video ─────────────────── #
        asset_map = {
            a.scene_number: a.file_path
            for a in state.generated_asset_pack.assets
            if a.status == "generated"
        }
        audio_map = {
            c.scene_number: c.audio_file_path
            for c in state.audio_plan.audio_cues
            if c.audio_file_path
        }

        scene_data = [
            {
                "image_path": asset_map.get(s.scene_number, ""),
                "audio_path": audio_map.get(s.scene_number),
                "duration":   s.duration_seconds,
            }
            for s in state.scene_breakdown.scenes
        ]

        safe_title = "".join(
            c if c.isalnum() or c in " _-" else "_"
            for c in (state.story_script.title if state.story_script else "animation")
        ).replace(" ", "_")

        output_path = str(VIDEO_DIR / f"{safe_title}.mp4")

        compose_video(
            scene_data=scene_data,
            output_path=output_path,
            fps=VIDEO_FPS,
            resolution=(VIDEO_WIDTH, VIDEO_HEIGHT),
        )

        state.output_video_path = output_path
        return state
