"""
Agent 1 — Prompt Understanding Agent

Input : raw user prompt (str)
Output: StructuredPrompt stored in state
"""
from core.base_agent import BaseAgent
from core.models import PipelineState, StructuredPrompt

SYSTEM_PROMPT = """You are an expert animation pre-production analyst.
Your job is to parse a natural language animation prompt and extract structured metadata.

Return ONLY a JSON object with this exact schema:
{
  "genre": "string",
  "tone": "string",
  "mood": "string",
  "target_audience": "string",
  "animation_style": "string (e.g. cartoon, anime, 3D, Pixar-like, stop-motion)",
  "duration_seconds": integer,
  "characters": [
    {
      "name": "string",
      "description": "string",
      "personality": "string",
      "visual_traits": "string"
    }
  ],
  "setting": "string",
  "themes": ["string"],
  "pacing": "string (slow / moderate / fast)"
}

Rules:
- Infer reasonable defaults when information is missing.
- duration_seconds should be 60–180 for short animations.
- animation_style default is "cartoon" if unspecified.
- Extract at least one character; create a narrator character if none are explicit.
"""


class PromptUnderstandingAgent(BaseAgent):
    name = "PromptUnderstandingAgent"
    color = "green"

    def _execute(self, state: PipelineState) -> PipelineState:
        raw = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=f"Parse this animation prompt:\n\n{state.user_prompt}",
        )
        data = self._extract_json(raw)
        state.structured_prompt = StructuredPrompt(**data)
        return state
