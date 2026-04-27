from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str
    description: str
    personality: str
    visual_traits: str


class StructuredPrompt(BaseModel):
    genre: str
    tone: str
    mood: str
    target_audience: str
    animation_style: str
    duration_seconds: int
    characters: List[Character]
    setting: str
    themes: List[str]
    pacing: str


class StoryScript(BaseModel):
    title: str
    beginning: str
    conflict: str
    resolution: str
    full_narrative: str
    emotional_arc: str
    word_count: int


class Scene(BaseModel):
    scene_number: int
    title: str
    duration_seconds: int
    description: str
    characters_present: List[str]
    actions: str
    camera_angle: str = "medium shot"
    setting: str = ""
    mood: str = "neutral"
    dialogue: Optional[str] = None


class SceneBreakdown(BaseModel):
    total_scenes: int
    total_duration_seconds: int
    scenes: List[Scene]


class VisualPrompt(BaseModel):
    scene_number: int
    image_generation_prompt: str
    style_notes: str = ""
    lighting: str = "natural lighting"
    color_palette: str = "vibrant"
    character_consistency_notes: str = ""
    environment_details: str = ""


class VisualPromptPack(BaseModel):
    animation_style: str
    global_style_guide: str
    prompts: List[VisualPrompt]


class GeneratedAsset(BaseModel):
    scene_number: int
    asset_type: str          # "image" | "video_clip"
    file_path: str           # placeholder path
    generation_model: str
    prompt_used: str
    status: str              # "generated" | "stub"
    notes: str = ""


class GeneratedAssetPack(BaseModel):
    assets: List[GeneratedAsset]
    total_scenes: int
    generation_summary: str


class AudioCue(BaseModel):
    scene_number: int
    narration_text: str
    voice_style: str = "neutral"
    timing_start_seconds: float = 0.0
    timing_end_seconds: float = 5.0
    sound_effects: List[str] = []
    audio_file_path: Optional[str] = None   # set by VoiceoverAudioAgent after TTS


class AudioPlan(BaseModel):
    narrator_voice: str
    background_music_genre: str
    background_music_tempo: str
    overall_audio_style: str
    audio_cues: List[AudioCue]
    music_transitions: List[str]


class SceneComposition(BaseModel):
    scene_number: int
    start_time_seconds: float
    end_time_seconds: float
    transition_in: str
    transition_out: str
    overlay_text: Optional[str] = None
    audio_sync_notes: str


class CompositionPlan(BaseModel):
    title: str
    total_duration_seconds: int
    frame_rate: int
    resolution: str
    scenes: List[SceneComposition]
    post_processing_notes: str
    export_format: str
    final_render_command: str


class QCIssue(BaseModel):
    severity: str            # "critical" | "warning" | "suggestion"
    category: str            # "continuity" | "audio_sync" | "visual" | "pacing"
    scene_reference: Optional[int] = None
    description: str
    suggested_fix: str


class QCReport(BaseModel):
    overall_score: float     # 0.0 – 10.0
    continuity_score: float
    audio_sync_score: float
    visual_consistency_score: float
    pacing_score: float
    issues: List[QCIssue]
    approved: bool
    improvement_summary: str


class PipelineState(BaseModel):
    user_prompt: str
    structured_prompt: Optional[StructuredPrompt] = None
    story_script: Optional[StoryScript] = None
    scene_breakdown: Optional[SceneBreakdown] = None
    visual_prompt_pack: Optional[VisualPromptPack] = None
    generated_asset_pack: Optional[GeneratedAssetPack] = None
    audio_plan: Optional[AudioPlan] = None
    composition_plan: Optional[CompositionPlan] = None
    qc_report: Optional[QCReport] = None
    qc_retry_count: int = 0
    status: str = "pending"
    output_video_path: Optional[str] = None   # set by AnimationCompositionAgent
    completed_stages: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
