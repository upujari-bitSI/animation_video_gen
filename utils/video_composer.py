"""
MoviePy-based video composer.

Takes a list of scene dicts (image_path, audio_path, duration) and
stitches them into a single MP4 with:
  • Ken Burns subtle zoom effect on each image
  • Fade-in on first scene, fade-out on last scene
  • Per-scene narration audio
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console

console = Console()


def compose_video(
    scene_data: List[dict],          # [{"image_path", "audio_path", "duration"}]
    output_path: str,
    fps: int = 24,
    resolution: Tuple[int, int] = (1280, 720),
    fade_duration: float = 0.5,
) -> str:
    """Compose all scenes into a final MP4. Returns output_path."""
    from moviepy.editor import (
        AudioFileClip,
        ImageClip,
        concatenate_videoclips,
    )
    from moviepy.video.fx.all import fadein, fadeout

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clips = []

    for i, scene in enumerate(scene_data):
        image_path = scene.get("image_path", "")
        audio_path = scene.get("audio_path") or ""
        duration   = max(float(scene.get("duration", 5)), 1.0)

        if not image_path or not Path(image_path).exists():
            console.print(f"  [yellow]Scene {i+1}: image not found ({image_path}), skipping[/yellow]")
            continue

        # ── Base clip ─────────────────────────────────────────────── #
        clip = ImageClip(image_path).set_duration(duration)
        clip = clip.resize(resolution)

        # Ken Burns: gentle 3 % zoom over the scene duration
        clip = clip.resize(lambda t, d=duration: 1 + 0.03 * t / d)

        # ── Audio ─────────────────────────────────────────────────── #
        if audio_path and Path(audio_path).exists():
            audio = AudioFileClip(audio_path)
            if audio.duration > duration:
                audio = audio.subclip(0, duration)
            clip = clip.set_audio(audio)

        # ── Transitions ───────────────────────────────────────────── #
        fd = min(fade_duration, duration / 4)
        if i == 0:
            clip = fadein(clip, fd)
        if i == len(scene_data) - 1:
            clip = fadeout(clip, fd)

        clips.append(clip)

    if not clips:
        raise RuntimeError("No valid scene clips to compose — check that images were generated.")

    console.print(f"  [cyan]Composing {len(clips)} scenes → {Path(output_path).name}[/cyan]")

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,          # suppress verbose moviepy progress bars
    )
    console.print(f"  [green bold]Video saved: {output_path}[/green bold]")
    return output_path
