"""
Agent 6 — Voiceover & Audio Agent

Input : StoryScript + SceneBreakdown
Output: AudioPlan stored in state

NOTE: In production, the narration_text for each cue would be passed to a
      TTS API (e.g. ElevenLabs, OpenAI TTS, Google Cloud TTS).
      Background music would be generated via MusicGen / Suno / Udio.
"""
from core.base_agent import BaseAgent
from core.models import AudioCue, AudioPlan, PipelineState

SYSTEM_PROMPT = """You are an audio director and sound designer for animated films.
Design a complete audio plan: narration, background music, and sound effects.

Return ONLY a JSON object with this exact schema:
{
  "narrator_voice": "string (voice description, e.g. warm female, excited child, wise elder)",
  "background_music_genre": "string",
  "background_music_tempo": "string (BPM description)",
  "overall_audio_style": "string",
  "audio_cues": [
    {
      "scene_number": integer,
      "narration_text": "string (exact words to narrate for this scene)",
      "voice_style": "string (e.g. gentle, excited, whispering)",
      "timing_start_seconds": float,
      "timing_end_seconds": float,
      "sound_effects": ["string"]
    }
  ],
  "music_transitions": ["string (describe music shifts between acts)"]
}

Audio design rules:
- Narration pacing must fit within each scene's duration.
- Sound effects should be vivid and scene-appropriate.
- Music should swell at emotional peaks and quiet during dialogue.
- The emotional arc of the music must mirror the story's arc.
- Each narration_text should be natural spoken language, not stage directions.
"""


class VoiceoverAudioAgent(BaseAgent):
    name = "VoiceoverAudioAgent"
    color = "magenta"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.story_script or not state.scene_breakdown:
            raise ValueError("StoryScript and SceneBreakdown required")

        scenes_data = [s.model_dump() for s in state.scene_breakdown.scenes]
        user_msg = (
            f"Design an audio plan for this animation.\n\n"
            f"Story title: {state.story_script.title}\n"
            f"Emotional arc: {state.story_script.emotional_arc}\n"
            f"Full narrative:\n{state.story_script.full_narrative}\n\n"
            f"Scene breakdown:\n{scenes_data}"
        )

        raw = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
        data = self._extract_json(raw)
        cues = [AudioCue(**c) for c in data["audio_cues"]]
        state.audio_plan = AudioPlan(
            narrator_voice=data["narrator_voice"],
            background_music_genre=data["background_music_genre"],
            background_music_tempo=data["background_music_tempo"],
            overall_audio_style=data["overall_audio_style"],
            audio_cues=cues,
            music_transitions=data["music_transitions"],
        )
        return state
