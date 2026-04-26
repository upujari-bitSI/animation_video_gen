"""
Agent 5 — Asset Generation Agent

Input : VisualPromptPack
Output: GeneratedAssetPack stored in state

NOTE: This agent is a structured stub.  In production, replace the
      `_generate_asset` method body with real calls to:
        - Image:      Stability AI / DALL·E / Midjourney API
        - Video clip: Sora / Runway Gen-3 / Pika Labs API
      The agent uses the LLM to plan the generation strategy and
      then simulates the asset creation with plausible file paths.
"""
from core.base_agent import BaseAgent
from core.models import GeneratedAsset, GeneratedAssetPack, PipelineState

SYSTEM_PROMPT = """You are an AI asset generation coordinator for animation production.
Given visual prompts for each scene, produce a structured asset generation plan
and simulate the output file paths that would be created.

Return ONLY a JSON object with this exact schema:
{
  "assets": [
    {
      "scene_number": integer,
      "asset_type": "image or video_clip",
      "file_path": "string (e.g. assets/scene_01.png or assets/scene_01.mp4)",
      "generation_model": "string (e.g. stable-diffusion-xl, dall-e-3, runway-gen3)",
      "prompt_used": "string (the final prompt sent to the model)",
      "status": "stub",
      "notes": "string (any generation notes)"
    }
  ],
  "total_scenes": integer,
  "generation_summary": "string"
}

Rules:
- Use video_clip for scenes with significant movement; image for static/pan scenes.
- Recommend the best model for each scene's complexity.
- Keep file_path convention: assets/scene_<zero-padded-number>.<ext>
"""


class AssetGenerationAgent(BaseAgent):
    name = "AssetGenerationAgent"
    color = "red"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.visual_prompt_pack:
            raise ValueError("VisualPromptPack required")

        prompts_data = [p.model_dump() for p in state.visual_prompt_pack.prompts]
        user_msg = (
            f"Plan asset generation for {len(prompts_data)} scenes.\n\n"
            f"Style: {state.visual_prompt_pack.animation_style}\n"
            f"Global guide: {state.visual_prompt_pack.global_style_guide}\n\n"
            f"Visual prompts:\n{prompts_data}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        assets = [GeneratedAsset(**a) for a in data["assets"]]
        state.generated_asset_pack = GeneratedAssetPack(
            assets=assets,
            total_scenes=data["total_scenes"],
            generation_summary=data["generation_summary"],
        )
        return state
