"""
Local TTS engine.

Default: pyttsx3  — zero download, uses the OS voice engine
                    (Windows SAPI5 / macOS NSSpeechSynthesizer / Linux espeak)
Optional: coqui   — higher quality; requires `pip install TTS`
                    first run downloads ~200 MB model
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def synthesize(
    text: str,
    output_path: str,
    engine: str = "pyttsx3",
    rate: int = 145,
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
) -> str:
    """
    Convert text to speech and save as a WAV file.
    Returns output_path on success, empty string on failure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        if engine == "coqui":
            return _coqui(text, output_path, coqui_model)
        return _pyttsx3(text, output_path, rate)
    except Exception as exc:
        console.print(f"  [yellow]TTS warning ({Path(output_path).name}): {exc}[/yellow]")
        return ""


# ── pyttsx3 ───────────────────────────────────────────────────────────────────

def _pyttsx3(text: str, output_path: str, rate: int) -> str:
    import pyttsx3

    # Re-init every call: Windows SAPI5 driver leaks state across save_to_file calls
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate",   rate)
        engine.setProperty("volume", 0.9)
        engine.save_to_file(text, str(Path(output_path).resolve()))
        engine.runAndWait()
    finally:
        engine.stop()
    return output_path


# ── Coqui TTS ─────────────────────────────────────────────────────────────────

def _coqui(text: str, output_path: str, model_name: str) -> str:
    from TTS.api import TTS   # noqa: PLC0415

    console.print(f"  [dim]Coqui TTS: loading {model_name}…[/dim]")
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path
