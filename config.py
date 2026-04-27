import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ── Image Generation (Stable Diffusion) ──────────────────────────────────────
# "runwayml/stable-diffusion-v1-5"  → ~4 GB VRAM  (default, recommended)
# "stabilityai/stable-diffusion-xl-base-1.0" → ~8 GB VRAM (higher quality)
SD_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
SD_DEVICE   = os.getenv("SD_DEVICE",   "auto")   # auto | cuda | cpu
SD_STEPS    = int(os.getenv("SD_STEPS",   "25"))
SD_WIDTH    = int(os.getenv("SD_WIDTH",  "768"))
SD_HEIGHT   = int(os.getenv("SD_HEIGHT", "512"))

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_ENGINE   = os.getenv("TTS_ENGINE",   "pyttsx3")  # pyttsx3 | coqui
COQUI_MODEL  = os.getenv("COQUI_MODEL",  "tts_models/en/ljspeech/tacotron2-DDC")
TTS_RATE     = int(os.getenv("TTS_RATE", "145"))     # words per minute

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
ASSETS_DIR = OUTPUT_DIR / "assets"
AUDIO_DIR  = OUTPUT_DIR / "audio"
VIDEO_DIR  = OUTPUT_DIR / "video"

# ── Video composition ─────────────────────────────────────────────────────────
VIDEO_FPS    = int(os.getenv("VIDEO_FPS",    "24"))
VIDEO_WIDTH  = int(os.getenv("VIDEO_WIDTH",  "1280"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "720"))

# ── Pipeline ──────────────────────────────────────────────────────────────────
MAX_QC_RETRIES = int(os.getenv("MAX_QC_RETRIES", "2"))

# Create output dirs on import
for _d in [ASSETS_DIR, AUDIO_DIR, VIDEO_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
