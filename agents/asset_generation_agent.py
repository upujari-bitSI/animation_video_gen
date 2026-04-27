"""
Agent 5 — Asset Generation Agent

Generates one PNG image per scene using a local Stable Diffusion model.
Images are saved to output/assets/scene_<N>.png.

To switch models, set SD_MODEL_ID in .env:
  - runwayml/stable-diffusion-v1-5      (default, ~4 GB VRAM)
  - stabilityai/stable-diffusion-xl-base-1.0  (higher quality, ~8 GB VRAM)
"""
from config import ASSETS_DIR, SD_DEVICE, SD_HEIGHT, SD_MODEL_ID, SD_STEPS, SD_WIDTH
from core.base_agent import BaseAgent
from core.models import GeneratedAsset, GeneratedAssetPack, PipelineState
from utils.image_generator import generate_image, resolve_device


class AssetGenerationAgent(BaseAgent):
    name = "AssetGenerationAgent"
    color = "red"

    def _execute(self, state: PipelineState) -> PipelineState:
        if not state.visual_prompt_pack:
            raise ValueError("VisualPromptPack required — run VisualPromptAgent first")

        device = resolve_device(SD_DEVICE)
        assets = []

        for vp in state.visual_prompt_pack.prompts:
            out_path = str(ASSETS_DIR / f"scene_{vp.scene_number:02d}.png")

            full_prompt = (
                f"{state.visual_prompt_pack.animation_style}, "
                f"{vp.image_generation_prompt}, "
                f"lighting: {vp.lighting}, "
                f"color palette: {vp.color_palette}, "
                f"high quality, detailed, professional animation, 4k"
            )

            try:
                generate_image(
                    prompt=full_prompt,
                    output_path=out_path,
                    model_id=SD_MODEL_ID,
                    device=device,
                    width=SD_WIDTH,
                    height=SD_HEIGHT,
                    steps=SD_STEPS,
                )
                status = "generated"
                notes = f"device={device}, steps={SD_STEPS}"
            except Exception as exc:
                status = "failed"
                notes = str(exc)
                out_path = ""

            assets.append(GeneratedAsset(
                scene_number=vp.scene_number,
                asset_type="image",
                file_path=out_path,
                generation_model=SD_MODEL_ID,
                prompt_used=full_prompt,
                status=status,
                notes=notes,
            ))

        generated_count = sum(1 for a in assets if a.status == "generated")
        state.generated_asset_pack = GeneratedAssetPack(
            assets=assets,
            total_scenes=len(assets),
            generation_summary=(
                f"Generated {generated_count}/{len(assets)} images "
                f"using {SD_MODEL_ID} on {device}"
            ),
        )
        return state
