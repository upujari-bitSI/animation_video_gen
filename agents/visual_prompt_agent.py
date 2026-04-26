"""
Agent 4 — Visual Prompt Agent

Input : SceneBreakdown + StructuredPrompt
Output: VisualPromptPack stored in state
"""
from core.base_agent import BaseAgent
from core.models import PipelineState, VisualPrompt, VisualPromptPack

SYSTEM_PROMPT = """You are an expert AI image/video generation prompt engineer
specialising in animation production.

Given a scene breakdown, produce precise visual prompts for each scene
to feed into image or video generation models (Stable Diffusion, DALL·E, Sora, etc.).

Return ONLY a JSON object with this exact schema:
{
  "animation_style": "string",
  "global_style_guide": "string (shared aesthetic rules for all scenes)",
  "prompts": [
    {
      "scene_number": integer,
      "image_generation_prompt": "string (detailed, comma-separated tags)",
      "style_notes": "string",
      "lighting": "string",
      "color_palette": "string",
      "character_consistency_notes": "string",
      "environment_details": "string"
    }
  ]
}

Prompt engineering rules:
- Start each prompt with the animation style keyword.
- Include: subject, action, environment, lighting, mood, art style, quality tags.
- Maintain consistent character descriptions across all scenes.
- Use negative-prompt style thinking to avoid inconsistencies.
- Add cinematic quality descriptors: 4K, detailed, professional animation.
"""


class VisualPromptAgent(BaseAgent):
    name = "VisualPromptAgent"
    color = "bright_yellow"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.scene_breakdown or not state.structured_prompt:
            raise ValueError("SceneBreakdown and StructuredPrompt required")

        scenes_data = [s.model_dump() for s in state.scene_breakdown.scenes]
        characters_data = [c.model_dump() for c in state.structured_prompt.characters]

        user_msg = (
            f"Generate visual prompts for each scene.\n\n"
            f"Animation style: {state.structured_prompt.animation_style}\n"
            f"Overall mood: {state.structured_prompt.mood}\n"
            f"Characters:\n{characters_data}\n\n"
            f"Scenes:\n{scenes_data}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        prompts = [VisualPrompt(**p) for p in data["prompts"]]
        state.visual_prompt_pack = VisualPromptPack(
            animation_style=data["animation_style"],
            global_style_guide=data["global_style_guide"],
            prompts=prompts,
        )
        return state
