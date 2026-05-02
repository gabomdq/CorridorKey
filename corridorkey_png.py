#!/usr/bin/env python3
"""
CorridorKey PNG -- Single-image chroma key removal.

Applies BiRefNet matting -> CorridorKey Inference -> Alpha composite.
Produces an RGBA PNG where the background has been removed (alpha = 0)
and the foreground RGB has been despilled by the CorridorKey inference
engine (same processing chain the video pipeline applies to every frame).

Usage:
    python corridorkey_png.py input.png [output.png] [options]

If output.png is omitted, the result is written next to the input as
<name>_keyed.png.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that the local packages
# (CorridorKeyModule, BiRefNetModule) are importable.
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ROCm env setup must happen before torch import
from device_utils import setup_rocm_env

setup_rocm_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("corridorkey_png")


# --- helpers -----------------------------------------------------------------

def _load_image_rgb(path: str) -> np.ndarray:
    """Load image as float32 sRGB [H, W, 3] in 0-1 range."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _save_rgba_png(path: str, rgb: np.ndarray, alpha: np.ndarray) -> None:
    """Save an RGBA PNG given sRGB [H,W,3] and alpha [H,W] or [H,W,1]."""
    if alpha.ndim == 3:
        alpha = alpha.squeeze(-1)
    rgba = np.dstack([rgb, alpha])
    # Convert 0-1 float -> 0-255 uint8
    rgba_u8 = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba_bgr = cv2.cvtColor(rgba_u8, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, rgba_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    log.info("Saved RGBA PNG -> %s", path)


# --- BiRefNet step (soft matte, no binary threshold) --------------------------

def run_birefnet_soft_matte(
    image_rgb: np.ndarray,
    device: str,
    usage: str = "Matting",
) -> np.ndarray:
    """
    Run BiRefNet matting model and preserve the full soft sigmoid matte
    (no binary thresholding). The soft fractional alpha values at edges
    give the downstream CorridorKey inference engine the information it
    needs to properly despill green spill.

    Returns alpha hint as float32 [0, 1] numpy array (H, W).
    """
    from BiRefNetModule.wrapper import BiRefNetHandler

    log.info("Loading BiRefNet (%s, soft matte) ...", usage)
    birefnet = BiRefNetHandler(device=device, usage=usage, soft_matte=True)

    # Convert float [0,1] -> uint8 BGR for BiRefNet's file-based API
    img_u8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        frame_path = os.path.join(tmp_in, "00000.png")
        cv2.imwrite(frame_path, img_bgr)

        log.info("Running BiRefNet pass (soft matte) ...")
        birefnet.process(
            input_path=tmp_in,
            alpha_output_dir=tmp_out,
            soft_matte=True,
        )

        # BiRefNet writes alphaSeq_00000.png
        alpha_path = os.path.join(tmp_out, "alphaSeq_00000.png")
        if not os.path.isfile(alpha_path):
            # Fallback: search for any .png
            for root, _dirs, files in os.walk(tmp_out):
                for f in files:
                    if f.endswith(".png"):
                        alpha_path = os.path.join(root, f)
                        break

        alpha_hint = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        if alpha_hint is None:
            raise RuntimeError(f"BiRefNet produced no output (checked {tmp_out})")

    birefnet.cleanup()
    del birefnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Soft matte values are in full 0-255 range (not binary). Convert to float [0,1].
    alpha_hint = alpha_hint.astype(np.float32) / 255.0

    log.info(
        "BiRefNet soft alpha hint: shape=%s, range=[%.4f, %.4f], mean=%.4f",
        alpha_hint.shape,
        alpha_hint.min(),
        alpha_hint.max(),
        alpha_hint.mean(),
    )
    return alpha_hint


# --- Inference step -----------------------------------------------------------

def run_inference_on_image(
    image_rgb: np.ndarray,
    alpha_hint: np.ndarray,
    device: str,
    screen_color: str,
    despill_strength: float,
    despeckle_size: int,
    refiner_scale: float,
    image_size: int,
    backend: str | None,
    input_is_linear: bool,
) -> dict:
    """
    Run the CorridorKey inference engine on image + BiRefNet soft matte.

    Returns the engine's output dict with keys 'alpha', 'fg', 'comp',
    'processed'.
    """
    from CorridorKeyModule.backend import create_engine

    log.info("Loading CorridorKey inference engine (color=%s, backend=%s, img_size=%d) ...",
             screen_color, backend or "auto", image_size)
    engine = create_engine(
        backend=backend,
        device=device,
        screen_color=screen_color,
        img_size=image_size,
    )

    # Ensure correct shape for engine: [H, W, 1]
    mask = alpha_hint.astype(np.float32)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]  # -> [H, W, 1]

    log.info("Running CorridorKey inference (despill=%.2f, despeckle=%d, refiner=%.2f) ...",
             despill_strength, despeckle_size, refiner_scale)
    result = engine.process_frame(
        image=image_rgb,           # [H, W, 3] sRGB 0-1
        mask_linear=mask,          # [H, W, 1] linear 0-1
        input_is_linear=input_is_linear,
        fg_is_straight=True,
        despill_strength=despill_strength,
        auto_despeckle=(despeckle_size > 0),
        despeckle_size=max(despeckle_size, 1),
        refiner_scale=refiner_scale,
        generate_comp=False,       # we don't need the checkerboard comp
        post_process_on_gpu=True,
    )

    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CorridorKey PNG -- Single-image chroma key removal"
    )
    parser.add_argument("input", help="Path to input PNG image")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path for output RGBA PNG (default: <input_stem>_keyed.png)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: cuda, mps, or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=("auto", "torch", "mlx"),
        help="Inference backend (default: auto-detect). MLX = Apple Silicon only.",
    )
    parser.add_argument(
        "--screen-color",
        default="green",
        choices=("green", "blue"),
        help="Screen color to key against (default: green)",
    )
    parser.add_argument(
        "--despill-strength",
        type=float,
        default=1.0,
        help="Despill strength 0.0-1.0 (default: 1.0). 1.0 = full despill.",
    )
    parser.add_argument(
        "--despeckle-size",
        type=int,
        default=400,
        help="Min connected-pixel area for alpha matte cleanup (default: 400)",
    )
    parser.add_argument(
        "--no-despeckle",
        action="store_true",
        help="Disable matte cleanup (despeckle). Equivalent to --despeckle-size 0.",
    )
    parser.add_argument(
        "--refiner-scale",
        type=float,
        default=1.0,
        help="Multiplier for refiner deltas (default: 1.0)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=2048,
        help="Inference image size used by the model (default: 2048)",
    )
    parser.add_argument(
        "--input-is-linear",
        action="store_true",
        help="Treat the input PNG as linear (instead of sRGB)",
    )
    parser.add_argument(
        "--birefnet-usage",
        default="Matting",
        metavar="USAGE",
        help="BiRefNet model variant (default: Matting). See BiRefNetModule docs for options.",
    )
    parser.add_argument(
        "--alpha-hint",
        default=None,
        help="Path to a pre-computed alpha hint PNG (bypasses BiRefNet)",
    )
    args = parser.parse_args()

    # --- Despeckle override ---------------------------------------------------
    despeckle_size = 0 if args.no_despeckle else args.despeckle_size

    # --- Resolve device -------------------------------------------------------
    from device_utils import resolve_device

    resolved_device = resolve_device(args.device)
    log.info("Using device: %s", resolved_device)

    # --- Load input -----------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Input file not found: {args.input}")

    image_rgb = _load_image_rgb(str(input_path))
    h, w = image_rgb.shape[:2]
    log.info("Loaded input image: %dx%d %s", w, h, input_path.name)

    # --- Alpha hint -----------------------------------------------------------
    if args.alpha_hint is not None:
        alpha_hint = cv2.imread(args.alpha_hint, cv2.IMREAD_GRAYSCALE)
        if alpha_hint is None:
            sys.exit(f"Cannot read alpha hint: {args.alpha_hint}")
        alpha_hint = alpha_hint.astype(np.float32) / 255.0
        log.info("Using provided alpha hint: %s", args.alpha_hint)
    else:
        alpha_hint = run_birefnet_soft_matte(image_rgb, resolved_device, usage=args.birefnet_usage)

    # --- CorridorKey inference ------------------------------------------------
    result = run_inference_on_image(
        image_rgb,
        alpha_hint,
        resolved_device,
        args.screen_color,
        args.despill_strength,
        despeckle_size,
        args.refiner_scale,
        args.image_size,
        args.backend,
        args.input_is_linear,
    )

    # --- Build final RGBA -----------------------------------------------------
    # result["alpha"] is the inference-refined matte [H, W, 1] (linear, 0-1)
    predicted_alpha = result["alpha"].squeeze(-1)  # -> [H, W]

    # Use the despilled foreground (sRGB) from the CorridorKey inference
    # engine -- this is the same FG the video pipeline writes to Output/FG/.
    # The engine removes screen-color spill from the foreground before
    # returning it, so background pixels are naturally dark and the subject
    # has no green/blue fringing in the output.
    despilled_fg = result["fg"]  # [H, W, 3] sRGB float 0-1

    output_path = args.output or str(input_path.with_suffix("")) + "_keyed.png"
    _save_rgba_png(output_path, despilled_fg, predicted_alpha)
    log.info("Done.")


if __name__ == "__main__":
    main()