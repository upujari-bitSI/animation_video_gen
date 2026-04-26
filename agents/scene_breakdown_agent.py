"""
Agent 3 — Scene Breakdown Agent

Input : StoryScript + StructuredPrompt
Output: SceneBreakdown stored in state
"""
from core.base_agent import BaseAgent
from core.models import PipelineState, Scene, SceneBreakdown

SYSTEM_PROMPT = """You are an animation director and storyboard artist.
Break down the given story script into individual scenes for production.

Return ONLY a JSON object with this exact schema:
{
  "total_scenes": integer,
  "total_duration_seconds": integer,
  "scenes": [
    {
      "scene_number": integer,
      "title": "string",
      "duration_seconds": integer (5-15 per scene),
      "description": "string (visual description of what's happening)",
      "characters_present": ["string"],
      "actions": "string (what each character does)",
      "camera_angle": "string (wide shot / close-up / medium / bird's eye / POV / etc.)",
      "setting": "string",
      "mood": "string",
      "dialogue": "string or null"
    }
  ]
}

Rules:
- Each scene should be 5–15 seconds.
- Total duration must match the target animation duration.
- Cover all plot points (beginning, conflict, resolution).
- Vary camera angles for visual dynamism.
- Include an establishing shot as scene 1.
- End with a satisfying final scene.
"""


class SceneBreakdownAgent(BaseAgent):
    name = "SceneBreakdownAgent"
    color = "yellow"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.story_script or not state.structured_prompt:
            raise ValueError("StoryScript and StructuredPrompt required")

        user_msg = (
            f"Break down this story into scenes.\n\n"
            f"Target duration: {state.structured_prompt.duration_seconds} seconds\n"
            f"Animation style: {state.structured_prompt.animation_style}\n"
            f"Story title: {state.story_script.title}\n\n"
            f"Full narrative:\n{state.story_script.full_narrative}\n\n"
            f"Characters: {[c.model_dump() for c in state.structured_prompt.characters]}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        scenes = [Scene(**s) for s in data["scenes"]]
        state.scene_breakdown = SceneBreakdown(
            total_scenes=len(scenes),
            total_duration_seconds=sum(s.duration_seconds for s in scenes),
            scenes=scenes,
        )
        return state
