from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str = "Unknown"
    description: str = ""
    personality: str = ""
    visual_traits: str = ""


class StructuredPrompt(BaseModel):
    genre: str = "adventure"
    tone: str = "uplifting"
    mood: str = "hopeful"
    target_audience: str = "general"
    animation_style: str = "cartoon"
    duration_seconds: int = 90
    characters: List[Character] = Field(default_factory=list)
    setting: str = ""
    themes: List[str] = Field(default_factory=list)
    pacing: str = "moderate"


class StoryScript(BaseModel):
    title: str = "Untitled"
    beginning: str = ""
    conflict: str = ""
    resolution: str = ""
    full_narrative: str = ""
    emotional_arc: str = ""
    word_count: int = 0


class Scene(BaseModel):
    scene_number: int
    title: str = ""
    duration_seconds: int = 10
    description: str = ""
    characters_present: List[str] = Field(default_factory=list)
    actions: str = ""
    camera_angle: str = "medium shot"
    setting: str = ""
    mood: str = "neutral"
    dialogue: Optional[str] = None


class SceneBreakdown(BaseModel):
    total_scenes: int = 0
    total_duration_seconds: int = 0
    scenes: List[Scene] = Field(default_factory=list)


class VisualPrompt(BaseModel):
    scene_number: int
    image_generation_prompt: str = ""
    style_notes: str = ""
    lighting: str = "natural lighting"
    color_palette: str = "vibrant"
    character_consistency_notes: str = ""
    environment_details: str = ""


class VisualPromptPack(BaseModel):
    animation_style: str = "cartoon"
    global_style_guide: str = ""
    prompts: List[VisualPrompt] = Field(default_factory=list)


class GeneratedAsset(BaseModel):
    scene_number: int
    asset_type: str = "image"
    file_path: str = ""
    generation_model: str = ""
    prompt_used: str = ""
    status: str = "stub"
    notes: str = ""


class GeneratedAssetPack(BaseModel):
    assets: List[GeneratedAsset] = Field(default_factory=list)
    total_scenes: int = 0
    generation_summary: str = ""


class AudioCue(BaseModel):
    scene_number: int
    narration_text: str = ""
    voice_style: str = "neutral"
    timing_start_seconds: float = 0.0
    timing_end_seconds: float = 5.0
    sound_effects: List[str] = Field(default_factory=list)
    audio_file_path: Optional[str] = None


class AudioPlan(BaseModel):
    narrator_voice: str = "neutral"
    background_music_genre: str = "ambient"
    background_music_tempo: str = "moderate"
    overall_audio_style: str = ""
    audio_cues: List[AudioCue] = Field(default_factory=list)
    music_transitions: List[str] = Field(default_factory=list)


class SceneComposition(BaseModel):
    scene_number: int
    start_time_seconds: float = 0.0
    end_time_seconds: float = 10.0
    transition_in: str = "fade"
    transition_out: str = "fade"
    overlay_text: Optional[str] = None
    audio_sync_notes: str = ""


class CompositionPlan(BaseModel):
    title: str = "Animation"
    total_duration_seconds: int = 90
    frame_rate: int = 24
    resolution: str = "1280x720"
    scenes: List[SceneComposition] = Field(default_factory=list)
    post_processing_notes: str = ""
    export_format: str = "mp4/h264"
    final_render_command: str = ""


class QCIssue(BaseModel):
    severity: str = "warning"
    category: str = "general"
    scene_reference: Optional[int] = None
    description: str = ""
    suggested_fix: str = ""


class QCReport(BaseModel):
    overall_score: float = 7.0
    continuity_score: float = 7.0
    audio_sync_score: float = 7.0
    visual_consistency_score: float = 7.0
    pacing_score: float = 7.0
    issues: List[QCIssue] = Field(default_factory=list)
    approved: bool = True
    improvement_summary: str = ""


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
    output_video_path: Optional[str] = None
    completed_stages: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
