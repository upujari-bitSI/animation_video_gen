"""
Agent 2 — Story Generation Agent

Input : StructuredPrompt
Output: StoryScript stored in state
"""
from core.base_agent import BaseAgent
from core.models import PipelineState, StoryScript

SYSTEM_PROMPT = """You are a master storyteller and animation scriptwriter.
Given structured metadata about an animation, write a complete short-film story script.

Return ONLY a JSON object with this exact schema:
{
  "title": "string",
  "beginning": "string (2-3 paragraphs)",
  "conflict": "string (2-3 paragraphs)",
  "resolution": "string (2-3 paragraphs)",
  "full_narrative": "string (complete story, 300-1000 words)",
  "emotional_arc": "string (describe the emotional journey)",
  "word_count": integer
}

Story guidelines:
- Create a compelling 3-act narrative arc.
- Match tone and audience from the metadata.
- Ensure characters are consistent with their defined personalities.
- For children's content, keep language simple and morals clear.
- Build emotional engagement through sensory details.
"""


class StoryGenerationAgent(BaseAgent):
    name = "StoryGenerationAgent"
    color = "blue"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.structured_prompt:
            raise ValueError("StructuredPrompt missing — run PromptUnderstandingAgent first")

        sp = state.structured_prompt
        user_msg = (
            f"Write a story script using this metadata:\n"
            f"Genre: {sp.genre}\n"
            f"Tone: {sp.tone}\n"
            f"Mood: {sp.mood}\n"
            f"Target Audience: {sp.target_audience}\n"
            f"Animation Style: {sp.animation_style}\n"
            f"Duration: ~{sp.duration_seconds} seconds\n"
            f"Setting: {sp.setting}\n"
            f"Themes: {', '.join(sp.themes)}\n"
            f"Pacing: {sp.pacing}\n"
            f"Characters: {[c.model_dump() for c in sp.characters]}\n"
            f"Original prompt: {state.user_prompt}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        data["word_count"] = len(data.get("full_narrative", "").split())
        state.story_script = StoryScript(**data)
        return state
