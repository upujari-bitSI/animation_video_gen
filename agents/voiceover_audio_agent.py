"""
Agent 6 — Voiceover & Audio Agent

Step 1: Uses the local LLM to design the full audio plan
        (narration text per scene, voice style, music cues).
Step 2: Synthesizes each scene's narration to a WAV file using
        local TTS (pyttsx3 by default, Coqui optional).

Audio files saved to: output/audio/scene_<N>_narration.wav
"""
from config import AUDIO_DIR, COQUI_MODEL, TTS_ENGINE, TTS_RATE
from core.base_agent import BaseAgent
from core.models import AudioCue, AudioPlan, PipelineState
from utils.tts_engine import synthesize

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
- Narration pacing must fit within each scene's duration (words per second ~2.5).
- Sound effects should be vivid and scene-appropriate.
- Music should swell at emotional peaks and quiet during dialogue.
- Each narration_text should be natural spoken language, not stage directions.
"""


class VoiceoverAudioAgent(BaseAgent):
    name = "VoiceoverAudioAgent"
    color = "magenta"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.story_script or not state.scene_breakdown:
            raise ValueError("StoryScript and SceneBreakdown required")

        # ── Step 1: LLM designs the audio plan ──────────────────────── #
        scenes_data = [s.model_dump() for s in state.scene_breakdown.scenes]
        user_msg = (
            f"Design an audio plan for this animation.\n\n"
            f"Story title: {state.story_script.title}\n"
            f"Emotional arc: {state.story_script.emotional_arc}\n"
            f"Full narrative:\n{state.story_script.full_narrative}\n\n"
            f"Scene breakdown:\n{scenes_data}"
        )

        raw  = self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=user_msg)
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

        # ── Step 2: Synthesise narration with local TTS ──────────────── #
        for cue in state.audio_plan.audio_cues:
            if not cue.narration_text.strip():
                continue
            audio_path = str(AUDIO_DIR / f"scene_{cue.scene_number:02d}_narration.wav")
            result = synthesize(
                text=cue.narration_text,
                output_path=audio_path,
                engine=TTS_ENGINE,
                rate=TTS_RATE,
                coqui_model=COQUI_MODEL,
            )
            cue.audio_file_path = result if result else None

        synthesised = sum(1 for c in state.audio_plan.audio_cues if c.audio_file_path)
        console_msg = f"  Synthesised {synthesised}/{len(state.audio_plan.audio_cues)} narration files"
        from rich.console import Console
        Console().print(f"  [dim]{console_msg}[/dim]")

        return state
