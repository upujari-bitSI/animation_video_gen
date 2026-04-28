from __future__ import annotations
from typing import Annotated, Any, List, Optional
from pydantic import BaseModel, BeforeValidator, Field


# ── Flexible coercion helpers ─────────────────────────────────────────────────
# llama3 sometimes returns [] or null for string fields, or a string for list
# fields. These validators silently coerce whatever the LLM returns.

def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(i) for i in v if i is not None)
    return str(v)

def _to_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(i) for i in v if i is not None]
    return []

def _to_int(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0

def _to_float(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0

def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() not in ("false", "0", "no", "")
    return bool(v)

# Annotated types — drop-in replacements for str / int / float / bool / List[str]
FlexStr      = Annotated[str,       BeforeValidator(_to_str)]
FlexInt      = Annotated[int,       BeforeValidator(_to_int)]
FlexFloat    = Annotated[float,     BeforeValidator(_to_float)]
FlexBool     = Annotated[bool,      BeforeValidator(_to_bool)]
FlexStrList  = Annotated[List[str], BeforeValidator(_to_str_list)]


# ── Data models ───────────────────────────────────────────────────────────────

class Character(BaseModel):
    name:         FlexStr = "Unknown"
    description:  FlexStr = ""
    personality:  FlexStr = ""
    visual_traits: FlexStr = ""


class StructuredPrompt(BaseModel):
    genre:            FlexStr    = "adventure"
    tone:             FlexStr    = "uplifting"
    mood:             FlexStr    = "hopeful"
    target_audience:  FlexStr    = "general"
    animation_style:  FlexStr    = "cartoon"
    duration_seconds: FlexInt    = 90
    characters:       List[Character] = Field(default_factory=list)
    setting:          FlexStr    = ""
    themes:           FlexStrList = Field(default_factory=list)
    pacing:           FlexStr    = "moderate"


class StoryScript(BaseModel):
    title:          FlexStr = "Untitled"
    beginning:      FlexStr = ""
    conflict:       FlexStr = ""
    resolution:     FlexStr = ""
    full_narrative: FlexStr = ""
    emotional_arc:  FlexStr = ""
    word_count:     FlexInt = 0


class Scene(BaseModel):
    scene_number:       FlexInt     = 0
    title:              FlexStr     = ""
    duration_seconds:   FlexInt     = 10
    description:        FlexStr     = ""
    characters_present: FlexStrList = Field(default_factory=list)
    actions:            FlexStr     = ""
    camera_angle:       FlexStr     = "medium shot"
    setting:            FlexStr     = ""
    mood:               FlexStr     = "neutral"
    dialogue:           Optional[FlexStr] = None


class SceneBreakdown(BaseModel):
    total_scenes:           FlexInt = 0
    total_duration_seconds: FlexInt = 0
    scenes: List[Scene] = Field(default_factory=list)


class VisualPrompt(BaseModel):
    scene_number:               FlexInt = 0
    image_generation_prompt:    FlexStr = ""
    style_notes:                FlexStr = ""
    lighting:                   FlexStr = "natural lighting"
    color_palette:              FlexStr = "vibrant"
    character_consistency_notes: FlexStr = ""
    environment_details:        FlexStr = ""


class VisualPromptPack(BaseModel):
    animation_style:   FlexStr = "cartoon"
    global_style_guide: FlexStr = ""
    prompts: List[VisualPrompt] = Field(default_factory=list)


class GeneratedAsset(BaseModel):
    scene_number:    FlexInt = 0
    asset_type:      FlexStr = "image"
    file_path:       FlexStr = ""
    generation_model: FlexStr = ""
    prompt_used:     FlexStr = ""
    status:          FlexStr = "stub"
    notes:           FlexStr = ""


class GeneratedAssetPack(BaseModel):
    assets: List[GeneratedAsset] = Field(default_factory=list)
    total_scenes:       FlexInt = 0
    generation_summary: FlexStr = ""


class AudioCue(BaseModel):
    scene_number:          FlexInt   = 0
    narration_text:        FlexStr   = ""
    voice_style:           FlexStr   = "neutral"
    timing_start_seconds:  FlexFloat = 0.0
    timing_end_seconds:    FlexFloat = 5.0
    sound_effects:         FlexStrList = Field(default_factory=list)
    audio_file_path:       Optional[str] = None


class AudioPlan(BaseModel):
    narrator_voice:          FlexStr     = "neutral"
    background_music_genre:  FlexStr     = "ambient"
    background_music_tempo:  FlexStr     = "moderate"
    overall_audio_style:     FlexStr     = ""
    audio_cues: List[AudioCue] = Field(default_factory=list)
    music_transitions: FlexStrList = Field(default_factory=list)


class SceneComposition(BaseModel):
    scene_number:        FlexInt   = 0
    start_time_seconds:  FlexFloat = 0.0
    end_time_seconds:    FlexFloat = 10.0
    transition_in:       FlexStr   = "fade"
    transition_out:      FlexStr   = "fade"
    overlay_text:        Optional[FlexStr] = None
    audio_sync_notes:    FlexStr   = ""


class CompositionPlan(BaseModel):
    title:                  FlexStr = "Animation"
    total_duration_seconds: FlexInt = 90
    frame_rate:             FlexInt = 24
    resolution:             FlexStr = "1280x720"
    scenes: List[SceneComposition] = Field(default_factory=list)
    post_processing_notes:  FlexStr = ""
    export_format:          FlexStr = "mp4/h264"
    final_render_command:   FlexStr = ""


class QCIssue(BaseModel):
    severity:        FlexStr = "warning"
    category:        FlexStr = "general"
    scene_reference: Optional[FlexInt] = None
    description:     FlexStr = ""
    suggested_fix:   FlexStr = ""


class QCReport(BaseModel):
    overall_score:              FlexFloat = 7.0
    continuity_score:           FlexFloat = 7.0
    audio_sync_score:           FlexFloat = 7.0
    visual_consistency_score:   FlexFloat = 7.0
    pacing_score:               FlexFloat = 7.0
    issues: List[QCIssue] = Field(default_factory=list)
    approved:            FlexBool = True
    improvement_summary: FlexStr  = ""


class PipelineState(BaseModel):
    user_prompt:         str
    structured_prompt:   Optional[StructuredPrompt]   = None
    story_script:        Optional[StoryScript]         = None
    scene_breakdown:     Optional[SceneBreakdown]      = None
    visual_prompt_pack:  Optional[VisualPromptPack]    = None
    generated_asset_pack: Optional[GeneratedAssetPack] = None
    audio_plan:          Optional[AudioPlan]           = None
    composition_plan:    Optional[CompositionPlan]     = None
    qc_report:           Optional[QCReport]            = None
    qc_retry_count:      int = 0
    status:              str = "pending"
    output_video_path:   Optional[str] = None
    completed_stages:    List[str] = Field(default_factory=list)
    errors:              List[str] = Field(default_factory=list)
