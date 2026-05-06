#!/usr/bin/env python3
"""CorridorKey -- single-input chroma key removal (image or video).

Image input (.png/.jpg/.jpeg/.webp):
    BiRefNet matting  →  CorridorKey inference  →  RGBA PNG (background = 0).

Video input (.mp4/.mov/.mkv/.avi/.webm):
    Per-frame  BiRefNet matting  →  CorridorKey inference  →  VP9 WebM with
    alpha (yuva420p), encoded streaming via ffmpeg stdin.  Optional scale,
    crop, frame-range trimming, framerate override, lossless or CRF VBR.

Usage:
    python scripts/corridorkey_png.py input.png    [output.png]   [options]
    python scripts/corridorkey_png.py input.mp4    [output.webm]  [options]

If the output path is omitted, the result is written next to the input as
``<stem>_keyed.<ext>``.
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

# Make sure the project root is on sys.path so that the local packages
# (CorridorKeyModule, BiRefNetModule) are importable.  Script lives in
# scripts/ so the project root is one directory up.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ROCm env setup must happen before torch import.
from device_utils import resolve_device, setup_rocm_env  # noqa: E402

setup_rocm_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("corridorkey")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


# --- I/O helpers -------------------------------------------------------------

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
    rgba_u8 = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba_bgr = cv2.cvtColor(rgba_u8, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, rgba_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    log.info("Saved RGBA PNG -> %s", path)


def _pack_rgba_u8(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pack [H,W,3] sRGB float + [H,W] (or [H,W,1]) alpha into [H,W,4] uint8."""
    if alpha.ndim == 3:
        alpha = alpha.squeeze(-1)
    rgba = np.dstack([rgb, alpha])
    return (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)


# --- BiRefNet -----------------------------------------------------------------

def _create_birefnet(device: str, usage: str):
    """Load BiRefNet once; the handler is reused for every frame."""
    from BiRefNetModule.wrapper import BiRefNetHandler

    log.info("Loading BiRefNet (%s, soft matte) ...", usage)
    handler = BiRefNetHandler(device=device, usage=usage, soft_matte=True)
    return handler


def _birefnet_alpha_for_frame(handler, image_rgb_f32: np.ndarray) -> np.ndarray:
    """Run BiRefNet on one frame.  Returns float32 alpha [H,W] in [0,1].

    Re-implements the per-frame body of ``BiRefNetHandler.process()`` so we
    can call the model without going through the file-based directory loop —
    necessary for low-latency streaming over a video sequence.  Soft matte
    only (no binary threshold, no dilate/erode).
    """
    from PIL import Image
    from torchvision import transforms

    from BiRefNetModule.wrapper import ImagePreprocessor, half_precision

    h, w = image_rgb_f32.shape[:2]
    img_u8 = (np.clip(image_rgb_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(img_u8)

    # Resolve dynamic-variant resolution on first call.
    resolution = handler.resolution
    if resolution is None:
        resolution = tuple(int(d) // 32 * 32 for d in pil_image.size)
        handler.resolution = resolution

    pre = ImagePreprocessor(resolution=tuple(resolution))
    inp = pre.proc(pil_image).unsqueeze(0).to(handler.device)
    if half_precision:
        inp = inp.half()

    with torch.no_grad():
        preds = handler.birefnet(inp)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred.float())
    mask = pred_pil.resize((w, h))
    return np.asarray(mask).astype(np.float32) / 255.0


# --- CorridorKey inference engine --------------------------------------------

def _create_engine(
    *, device: str, screen_color: str, image_size: int, backend: str | None,
):
    """Load the CorridorKey engine once; keep it for the whole input."""
    from CorridorKeyModule.backend import create_engine

    log.info(
        "Loading CorridorKey engine (color=%s, backend=%s, img_size=%d) ...",
        screen_color, backend or "auto", image_size,
    )
    return create_engine(
        backend=backend,
        device=device,
        screen_color=screen_color,
        img_size=image_size,
    )


def _run_engine(
    engine,
    image_rgb: np.ndarray,
    alpha_hint: np.ndarray,
    *,
    despill_strength: float,
    despeckle_size: int,
    refiner_scale: float,
    input_is_linear: bool,
    post_process_on_gpu: bool,
    screen_channel: int,
) -> dict:
    """Run a single inference call.  ``image_rgb`` is [H,W,3] sRGB float;
    ``alpha_hint`` is [H,W] float in [0,1]."""
    mask = alpha_hint.astype(np.float32)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    return engine.process_frame(
        image=image_rgb,
        mask_linear=mask,
        input_is_linear=input_is_linear,
        fg_is_straight=True,
        despill_strength=despill_strength,
        auto_despeckle=(despeckle_size > 0),
        despeckle_size=max(despeckle_size, 1),
        refiner_scale=refiner_scale,
        generate_comp=False,
        post_process_on_gpu=post_process_on_gpu,
        screen_channel=screen_channel,
    )


def _detect_screen_color(image_rgb: np.ndarray, alpha_hint: np.ndarray) -> str:
    """Probe a single (image, alpha) pair to pick green vs blue.

    Mirrors ``clip_manager._resolve_screen_color`` for "auto" — uses
    :func:`CorridorKeyModule.core.color_utils.estimate_screen_color`,
    which inspects pixels with ``alpha < 0.3`` (the screen background).
    """
    from CorridorKeyModule.core.color_utils import estimate_screen_color
    return estimate_screen_color(image_rgb, alpha_hint)


# --- On-disk cache ------------------------------------------------------------
#
# Re-runs with the same input but different output options (cut range, scale,
# crop, framerate, codec settings) shouldn't re-run BiRefNet/GVM or the
# CorridorKey engine.  We cache two artifacts per frame, both indexed by the
# global frame index, in a single shared cache folder so multiple inputs can
# coexist without separate top-level dirs:
#   <cache_root>/alphahint_<method>/<input_stem>_NNNNNN.png  uint8 alpha hint
#   <cache_root>/keyed/<input_stem>_NNNNNN.png              RGBA keyed frame
#
# When all keyed frames in the requested range are present, we skip model
# loading entirely and just re-encode.  When only the alpha hints are cached,
# we skip the alpha-generation pass.  When nothing is cached, we run the
# full pipeline and populate both layers.

FRAME_INDEX_FMT = "{:06d}"


def _resolve_cache_root(args: argparse.Namespace) -> Path | None:
    """Return the shared cache root, or None when caching is off.

    The same root is used by every input run from this working dir; the
    input stem is encoded in each cache file's name (see ``_frame_filename``)
    so multiple inputs share the folder safely.
    """
    if args.no_cache:
        return None
    base = Path(args.cache_dir) if args.cache_dir else Path.cwd()
    return base / "corridorkey_cache"


def _alphahint_cache_dir(cache_root: Path | None, method: str) -> Path | None:
    return None if cache_root is None else cache_root / f"alphahint_{method}"


def _keyed_cache_dir(cache_root: Path | None) -> Path | None:
    return None if cache_root is None else cache_root / "keyed"


def _frame_filename(input_stem: str, idx: int) -> str:
    """Cache-file name for one frame of one input.

    Format: ``<input_stem>_<6-digit-index>.png``.  The stem disambiguates
    files from different inputs sharing the same cache folder.
    """
    return f"{input_stem}_{FRAME_INDEX_FMT.format(idx)}.png"


def _frame_path(cache_dir: Path | None, input_stem: str, idx: int) -> Path | None:
    return None if cache_dir is None else cache_dir / _frame_filename(input_stem, idx)


def _all_present(cache_dir: Path | None, input_stem: str, start: int, end: int) -> bool:
    """True when every frame in ``[start, end)`` for ``input_stem`` is cached."""
    if cache_dir is None or not cache_dir.is_dir():
        return False
    return all(
        (cache_dir / _frame_filename(input_stem, i)).is_file()
        for i in range(start, end)
    )


def _clear_input_files(cache_root: Path | None, input_stem: str) -> int:
    """Remove every cached file belonging to ``input_stem`` from every
    subdir of ``cache_root``.  Other inputs' caches stay intact.
    Returns the count of files removed.
    """
    if cache_root is None or not cache_root.is_dir():
        return 0
    count = 0
    for sub in cache_root.iterdir():
        if sub.is_dir():
            for f in sub.glob(f"{input_stem}_*.png"):
                f.unlink()
                count += 1
    return count


def _read_cached_keyed(path: Path) -> np.ndarray:
    """Read a cached RGBA PNG as uint8 [H, W, 4] in RGBA order."""
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise RuntimeError(f"Cannot read cached keyed frame: {path}")
    if bgra.ndim == 3 and bgra.shape[2] == 4:
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    raise RuntimeError(f"Cached keyed frame is not RGBA: {path} (shape={bgra.shape})")


def _write_cached_keyed(path: Path, rgba_u8: np.ndarray) -> None:
    bgra = cv2.cvtColor(rgba_u8, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def _read_cached_alphahint(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read cached alpha hint: {path}")
    return m.astype(np.float32) / 255.0


def _write_cached_alphahint(path: Path, alpha_f32: np.ndarray) -> None:
    cv2.imwrite(str(path), (np.clip(alpha_f32, 0, 1) * 255).astype(np.uint8))


# --- BiRefNet batch (whole-clip pre-pass) ------------------------------------

def _generate_birefnet_masks(
    input_path: Path,
    output_dir: Path,
    input_stem: str,
    device: str,
    usage: str,
) -> list[Path]:
    """Run BiRefNet over every frame of ``input_path`` up-front.

    Same per-frame algorithm as the streaming path
    (:func:`_birefnet_alpha_for_frame`), but does the whole clip in one
    pass and writes the results to the cache.  Used when VideoMaMa is
    requested with ``--mask-hint birefnet`` and there are no cached
    BiRefNet hints yet for this input.
    """
    handler = _create_birefnet(device, usage)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Running BiRefNet over %d frames → %s ...", n_total, output_dir)

    written: list[Path] = []
    try:
        idx = 0
        last_log = time.monotonic()
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            alpha = _birefnet_alpha_for_frame(handler, frame_rgb)
            dst = output_dir / _frame_filename(input_stem, idx)
            _write_cached_alphahint(dst, alpha)
            written.append(dst)
            idx += 1
            now = time.monotonic()
            if now - last_log >= 5.0:
                log.info("BiRefNet: %d/%d frames", idx, n_total)
                last_log = now
    finally:
        cap.release()
        handler.cleanup()
        del handler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("BiRefNet wrote %d masks → %s", len(written), output_dir)
    return written


# --- VideoMaMa (mask-hint refinement) ----------------------------------------

def _find_cached_videomama_hint_source(
    cache_root: Path | None, input_stem: str,
) -> tuple[Path, str] | None:
    """Locate an existing alpha-hint cache for ``input_stem`` to feed VideoMaMa.

    Returns ``(dir, method_name)`` for the chosen source, or ``None`` if no
    cached hints exist for this input.  Prefers ``alphahint_gvm/`` (higher-
    quality coarse matte) over ``alphahint_birefnet/`` when both are present.
    """
    if cache_root is None:
        return None
    for method in ("gvm", "birefnet"):
        d = cache_root / f"alphahint_{method}"
        if d.is_dir() and any(d.glob(f"{input_stem}_*.png")):
            return d, method
    return None


def _read_mask_hint_frames(
    mask_hint_path: Path, *, stem_filter: str | None = None,
) -> list[np.ndarray]:
    """Load a coarse mask hint as a list of binary-thresholded uint8 frames.

    Mirrors the wizard's behaviour in ``clip_manager.run_videomama``: accepts
    either a directory of mask images (PNG/JPG/EXR) or a video file, force-
    thresholds to binary at the same level (>10).

    When ``stem_filter`` is given (used for the auto-detected cache source),
    only ``<stem_filter>_*.png`` files in the dir are picked up — this skips
    other inputs' caches that share the folder.
    """
    frames: list[np.ndarray] = []
    if mask_hint_path.is_dir():
        if stem_filter is not None:
            files = sorted(mask_hint_path.glob(f"{stem_filter}_*.png"))
        else:
            files = sorted(
                f for f in mask_hint_path.iterdir()
                if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".exr")
            )
        for f in files:
            if f.suffix.lower() == ".exr":
                m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if m is None:
                    continue
                if m.ndim == 3:
                    m = m[:, :, 0]
                m = (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
            _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
            frames.append(m)
    elif mask_hint_path.is_file():
        cap = cv2.VideoCapture(str(mask_hint_path))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                m = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
                frames.append(m)
        finally:
            cap.release()
    else:
        raise FileNotFoundError(f"--mask-hint not found: {mask_hint_path}")
    return frames


def _generate_videomama_masks(
    input_path: Path,
    mask_hint_path: Path,
    output_dir: Path,
    input_stem: str,
    device: str,
    chunk_size: int,
    *,
    hint_stem_filter: str | None = None,
) -> list[Path]:
    """Run VideoMaMa on (input video, coarse mask hint) → per-frame refined
    alpha masks in ``output_dir`` as ``<input_stem>_NNNNNN.png``.

    Mirrors the wizard's ``run_videomama`` (clip_manager.py) — same model
    loading pattern, same binary-threshold pre-processing on the mask hint,
    same chunked iteration.  Drops the pipeline from VRAM before returning.

    ``hint_stem_filter`` is set when the hint dir is the shared cache
    (``alphahint_<method>/``) so only this input's PNG files are picked up;
    when reading a user-supplied directory or video, leave it as ``None``.

    NOTE: VideoMaMa requires every input frame and mask frame to be loaded
    into RAM at once (the inference module's API takes lists, not a stream),
    so peak host memory is ~``n_frames * (img + mask)``.
    """
    # The VideoMaMaInferenceModule uses intra-package imports that assume its
    # own directory is on sys.path (mirrors clip_manager.run_videomama).
    sys.path.append(str(_project_root / "VideoMaMaInferenceModule"))
    from VideoMaMaInferenceModule.inference import (  # noqa: E402
        load_videomama_model,
        run_inference as run_videomama_frames,
    )

    log.info("Loading VideoMaMa pipeline on %s ...", device)
    pipeline = load_videomama_model(device=device)

    log.info("Reading input frames from %s ...", input_path)
    cap = cv2.VideoCapture(str(input_path))
    input_frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    log.info("Loaded %d input frames", len(input_frames))

    log.info("Reading mask hint from %s ...", mask_hint_path)
    mask_frames = _read_mask_hint_frames(mask_hint_path, stem_filter=hint_stem_filter)
    log.info("Loaded %d mask frames", len(mask_frames))

    n = min(len(input_frames), len(mask_frames))
    if n == 0:
        raise RuntimeError("VideoMaMa: no valid (input, mask) pairs to process.")
    if len(input_frames) != len(mask_frames):
        log.warning(
            "VideoMaMa: input has %d frames, mask hint has %d — using %d.",
            len(input_frames), len(mask_frames), n,
        )
    input_frames = input_frames[:n]
    mask_frames = mask_frames[:n]

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Running VideoMaMa (chunk_size=%d) over %d frames ...", chunk_size, n)
    written: list[Path] = []
    saved = 0
    for chunk in run_videomama_frames(pipeline, input_frames, mask_frames, chunk_size=chunk_size):
        for frame_rgb in chunk:
            if saved >= n:
                break
            # VideoMaMa returns the matte as a 3-channel RGB image (channels
            # are equal — it's a grayscale matte broadcast).  Take a single
            # channel to match BiRefNet/GVM's grayscale cache convention.
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            dst = output_dir / _frame_filename(input_stem, saved)
            cv2.imwrite(str(dst), gray, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            written.append(dst)
            saved += 1
        log.info("VideoMaMa: %d/%d frames written", saved, n)

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("VideoMaMa wrote %d masks → %s", len(written), output_dir)
    return written


# --- GVM (Generative Video Matting) ------------------------------------------

def _generate_gvm_masks(
    input_path: Path, output_dir: Path, input_stem: str, device: str,
) -> list[Path]:
    """Run GVM on ``input_path`` and write per-frame mask PNGs into
    ``output_dir`` using ``<input_stem>_NNNNNN.png`` names so the shared cache
    folder can hold output for multiple inputs.

    GVM is the auto-matte path used by the ``g`` action of the wizard
    (``clip_manager.generate_alphas``).  Mirrors its parameters — single-frame
    batches and a 1-step denoise so the diffusion stays fast and the output
    is per-frame deterministic enough for downstream chroma keying.  Drops
    the GVM model from VRAM before returning so the inference engine can
    be loaded next.

    GVM writes its own ``0001.png`` counter scheme; we capture into a tempdir
    and then move/rename into the shared cache with the stem-prefixed names.
    """
    from clip_manager import get_gvm_processor

    log.info("Loading GVM on %s ...", device)
    processor = get_gvm_processor(device=device)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gvm_raw_") as gvm_raw:
        log.info("Generating per-frame alpha masks via GVM ...")
        processor.process_sequence(
            input_path=str(input_path),
            output_dir=None,
            num_frames_per_batch=1,
            decode_chunk_size=1,
            denoise_steps=1,
            mode="matte",
            write_video=False,
            direct_output_dir=gvm_raw,
        )
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gvm_outputs = sorted(p for p in Path(gvm_raw).iterdir() if p.suffix.lower() == ".png")
        log.info("Renaming %d GVM masks → %s/", len(gvm_outputs), output_dir)
        renamed: list[Path] = []
        for i, src in enumerate(gvm_outputs):
            dst = output_dir / _frame_filename(input_stem, i)
            # GVM's tempdir is on /tmp (or wherever); cache is in cwd.
            # shutil.move handles cross-filesystem cases; falls back to copy+unlink.
            import shutil
            shutil.move(str(src), str(dst))
            renamed.append(dst)
    return renamed


# --- ffmpeg sink (video output) ---------------------------------------------

def _open_ffmpeg_rgba_sink(
    width: int, height: int, framerate: float, output_path: Path, args: argparse.Namespace,
) -> subprocess.Popen:
    """Spawn ffmpeg reading raw RGBA frames on stdin, writing VP9 yuva420p WebM.

    Scale and crop filters are applied on the ffmpeg side; both are optional.
    """
    vf_chain: list[str] = []
    if args.crop:
        vf_chain.append(f"crop={args.crop}")
    if args.scale:
        vf_chain.append(f"scale={args.scale}")

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", f"{framerate}",
        "-i", "-",
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-deadline", args.deadline,
        "-cpu-used", str(args.cpu_used),
    ]
    if args.crf is None:
        cmd += ["-lossless", "1"]
    else:
        cmd += ["-b:v", "0", "-crf", str(args.crf)]
    if vf_chain:
        cmd += ["-vf", ",".join(vf_chain)]
    cmd += ["-an", str(output_path)]

    log.info("ffmpeg sink: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# --- entry points ------------------------------------------------------------

def run_image(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    image_rgb = _load_image_rgb(str(input_path))
    h, w = image_rgb.shape[:2]
    log.info("Loaded image: %dx%d %s", w, h, input_path.name)

    if args.alpha_hint is not None:
        alpha_raw = cv2.imread(args.alpha_hint, cv2.IMREAD_GRAYSCALE)
        if alpha_raw is None:
            sys.exit(f"Cannot read alpha hint: {args.alpha_hint}")
        alpha_hint = alpha_raw.astype(np.float32) / 255.0
        if alpha_hint.shape[:2] != (h, w):
            alpha_hint = cv2.resize(alpha_hint, (w, h), interpolation=cv2.INTER_LINEAR)
        log.info("Using provided alpha hint: %s", args.alpha_hint)
    else:
        handler = _create_birefnet(args.device, args.birefnet_usage)
        try:
            alpha_hint = _birefnet_alpha_for_frame(handler, image_rgb)
            log.info(
                "BiRefNet soft alpha: shape=%s, range=[%.3f, %.3f], mean=%.3f",
                alpha_hint.shape, alpha_hint.min(), alpha_hint.max(), alpha_hint.mean(),
            )
        finally:
            handler.cleanup()
            del handler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.screen_color == "auto":
        args.screen_color = _detect_screen_color(image_rgb, alpha_hint)
        log.info("Auto-detected screen color: %s", args.screen_color)

    from CorridorKeyModule.core.color_utils import screen_channel_for_color
    screen_channel = screen_channel_for_color(args.screen_color)

    engine = _create_engine(
        device=args.device, screen_color=args.screen_color,
        image_size=args.image_size, backend=args.backend,
    )
    log.info("Running CorridorKey inference (despill=%.2f, despeckle=%d, refiner=%.2f) ...",
             args.despill_strength, args.despeckle_size, args.refiner_scale)
    result = _run_engine(
        engine, image_rgb, alpha_hint,
        despill_strength=args.despill_strength,
        despeckle_size=args.despeckle_size,
        refiner_scale=args.refiner_scale,
        input_is_linear=args.input_is_linear,
        post_process_on_gpu=args.gpu_post_processing,
        screen_channel=screen_channel,
    )
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _save_rgba_png(str(output_path), result["fg"], result["alpha"].squeeze(-1))


def _encode_from_keyed_cache(
    keyed_dir: Path,
    input_stem: str,
    *,
    start: int, n_frames: int,
    width: int, height: int, framerate: float,
    output_path: Path, args: argparse.Namespace,
) -> None:
    """Fast path used when every requested frame is already keyed on disk.

    Streams cached RGBA PNGs straight to ffmpeg with no model loads.
    """
    proc = _open_ffmpeg_rgba_sink(width, height, framerate, output_path, args)
    assert proc.stdin is not None
    t0 = time.monotonic()
    last_log = t0
    try:
        for i in range(n_frames):
            global_idx = start + i
            rgba_u8 = _read_cached_keyed(keyed_dir / _frame_filename(input_stem, global_idx))
            proc.stdin.write(rgba_u8.tobytes())
            now = time.monotonic()
            if now - last_log >= 5.0:
                done = i + 1
                elapsed = now - t0
                log.info("encoded %d/%d frames (%.1f fps)",
                         done, n_frames, done / elapsed)
                last_log = now
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = proc.wait()
        if rc != 0:
            sys.exit(f"ffmpeg exited with code {rc}")


def run_video(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    if args.alpha_hint is not None:
        sys.exit("--alpha-hint is image-only; for video, use --alpha-method instead.")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    framerate = args.framerate or fps
    start = max(0, args.start_frame or 0)
    end = min(n_total, args.end_frame) if args.end_frame else n_total
    if end <= start:
        sys.exit("--start-frame / --end-frame leaves no frames to process.")
    n_frames = end - start

    input_stem = input_path.stem
    cache_root = _resolve_cache_root(args)
    alpha_dir = _alphahint_cache_dir(cache_root, args.alpha_method)
    keyed_dir = _keyed_cache_dir(cache_root)
    if cache_root is not None:
        log.info("Cache root: %s (this input → %s_NNNNNN.png)", cache_root, input_stem)
    log.info(
        "Video: %dx%d @ %.3f fps, %d frames total, processing %d (%d..%d), method=%s",
        width, height, fps, n_total, n_frames, start, end - 1, args.alpha_method,
    )

    # Fast path: every requested frame is already keyed on disk → encode only.
    if _all_present(keyed_dir, input_stem, start, end):
        log.info(
            "FAST PATH: all %d keyed frames cached for %r — skipping alpha + "
            "inference, encode-only. Pass --clean to invalidate this cache "
            "and re-run inference (necessary when --despill-strength, "
            "--refiner-scale, --image-size, --screen-color, or "
            "--gpu-post-processing change).",
            n_frames, input_stem,
        )
        _encode_from_keyed_cache(
            keyed_dir,  # type: ignore[arg-type]  # _all_present guarantees not None
            input_stem,
            start=start, n_frames=n_frames,
            width=width, height=height, framerate=framerate,
            output_path=output_path, args=args,
        )
        log.info("Wrote %s", output_path)
        return

    # Some keyed frames missing → we need the inference engine (and possibly
    # the alpha generator).  For GVM, masks are batch-generated for the whole
    # video; if ANY mask is missing we regenerate them all (this input's files
    # only — other inputs in the shared cache are left intact).  For BiRefNet,
    # alpha generation is per-frame so partial caches are fine.
    if args.alpha_method in ("gvm", "videomama"):

        def _clear_stale_for_input() -> None:
            if alpha_dir is None or not alpha_dir.is_dir():
                return
            stale = list(alpha_dir.glob(f"{input_stem}_*.png"))
            if stale:
                log.info(
                    "%s cache for %r incomplete; clearing %d stale files",
                    args.alpha_method.upper(), input_stem, len(stale),
                )
                for f in stale:
                    f.unlink()

        def _run_alpha_gen(target_dir: Path) -> None:
            if args.alpha_method == "gvm":
                _generate_gvm_masks(input_path, target_dir, input_stem, args.device)
                return

            # videomama needs a coarse mask hint.  Three ways to supply it:
            #   1) --mask-hint birefnet|gvm  → use that method's cached hints,
            #      and if they aren't present yet, run that method first to
            #      populate the cache (or a tempdir, if caching is off).
            #   2) --mask-hint <path>        → external dir or video file.
            #   3) --mask-hint omitted       → auto-detect from cache
            #      (alphahint_gvm preferred, alphahint_birefnet fallback;
            #      error out if neither exists).
            hint_method: str | None = (
                args.mask_hint if args.mask_hint in ("birefnet", "gvm") else None
            )

            if hint_method:
                # Method-name path: cache subdir for this method (or tempdir
                # when --no-cache).  Generate first if not present yet.
                if cache_root is not None:
                    hint_dir = cache_root / f"alphahint_{hint_method}"
                else:
                    tmp_hints = tempfile.TemporaryDirectory(prefix="corridorkey_hints_")
                    hint_dir = Path(tmp_hints.name)
                    args._tmp_hint_holder = tmp_hints  # keep alive for this run

                if not _all_present(hint_dir, input_stem, 0, n_total):
                    log.info(
                        "VideoMaMa: %s hints not cached for %r; running %s first.",
                        hint_method, input_stem, hint_method,
                    )
                    if hint_method == "birefnet":
                        _generate_birefnet_masks(
                            input_path, hint_dir, input_stem,
                            args.device, args.birefnet_usage,
                        )
                    else:  # gvm
                        _generate_gvm_masks(
                            input_path, hint_dir, input_stem, args.device,
                        )
                else:
                    log.info("VideoMaMa: refining cached %s hints.", hint_method)

                hint_path = hint_dir
                hint_stem_filter: str | None = input_stem

            elif args.mask_hint:
                hint_path = Path(args.mask_hint)
                hint_stem_filter = None
                log.info("VideoMaMa: using user-supplied mask hint: %s", hint_path)

            else:
                found = _find_cached_videomama_hint_source(cache_root, input_stem)
                if found is None:
                    sys.exit(
                        f"VideoMaMa needs a coarse alpha hint to refine, but none was "
                        f"found for {input_stem!r}. Either:\n"
                        f"  1) Pass --mask-hint birefnet (or gvm) to auto-generate the "
                        f"hints first, then refine.\n"
                        f"  2) Run --alpha-method=birefnet or --alpha-method=gvm first "
                        f"to populate the cache, then re-run with --alpha-method=videomama.\n"
                        f"  3) Pass --mask-hint <video-or-dir> with an externally produced hint."
                    )
                hint_path, hint_method = found
                hint_stem_filter = input_stem
                log.info(
                    "VideoMaMa: refining cached %s alpha hints from %s",
                    hint_method, hint_path,
                )

            _generate_videomama_masks(
                input_path,
                hint_path,
                target_dir,
                input_stem,
                args.device,
                args.videomama_chunk_size,
                hint_stem_filter=hint_stem_filter,
            )

        if alpha_dir is not None and not _all_present(alpha_dir, input_stem, 0, n_total):
            _clear_stale_for_input()
            _run_alpha_gen(alpha_dir)
        elif alpha_dir is None:
            # Caching disabled: write masks to a tempdir for this run only.
            tmp_holder = tempfile.TemporaryDirectory(prefix="corridorkey_alpha_")
            alpha_dir = Path(tmp_holder.name)
            _run_alpha_gen(alpha_dir)
            args._tmp_alpha_holder = tmp_holder  # keep alive until end of run

    # BiRefNet handler is created lazily on the first uncached alpha hint.
    handler = None

    def alpha_provider(idx: int, frame_rgb: np.ndarray) -> np.ndarray:
        nonlocal handler
        h, w = frame_rgb.shape[:2]
        cached = _frame_path(alpha_dir, input_stem, idx)
        if cached and cached.is_file():
            alpha = _read_cached_alphahint(cached)
        elif args.alpha_method in ("gvm", "videomama"):
            raise RuntimeError(
                f"{args.alpha_method.upper()} mask missing for frame {idx} of "
                f"{input_stem!r} after generation pass — check cache dir {alpha_dir}"
            )
        else:
            if handler is None:
                handler = _create_birefnet(args.device, args.birefnet_usage)
            alpha = _birefnet_alpha_for_frame(handler, frame_rgb)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                _write_cached_alphahint(cached, alpha)
        # GVM masks are saved at GVM's processing resolution (e.g. 576p on an
        # 8 GiB card), which doesn't match the input video's native size.
        # Mirror clip_manager.run_inference: resize the alpha to match the
        # frame's H×W before handing to the engine.
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return alpha

    # Auto-detect screen color: probe frame `start` and its alpha hint.
    # alpha_provider populates the cache on first call so this work isn't
    # wasted — the alpha is reused inside the main loop.
    if args.screen_color == "auto":
        cap_probe = cv2.VideoCapture(str(input_path))
        cap_probe.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret_probe, probe_bgr = cap_probe.read()
        cap_probe.release()
        if not ret_probe:
            log.warning("Auto screen-color: cannot read frame %d, defaulting to green.", start)
            args.screen_color = "green"
        else:
            probe_rgb = cv2.cvtColor(probe_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            probe_alpha = alpha_provider(start, probe_rgb)
            args.screen_color = _detect_screen_color(probe_rgb, probe_alpha)
            log.info("Auto-detected screen color: %s", args.screen_color)

    from CorridorKeyModule.core.color_utils import screen_channel_for_color
    screen_channel = screen_channel_for_color(args.screen_color)

    engine = _create_engine(
        device=args.device, screen_color=args.screen_color,
        image_size=args.image_size, backend=args.backend,
    )

    if keyed_dir is not None:
        keyed_dir.mkdir(parents=True, exist_ok=True)

    # Report cache breakdown so it's obvious how many frames will hit
    # inference vs. be re-used from the keyed cache.
    if keyed_dir is not None and keyed_dir.is_dir():
        cached_count = sum(
            (keyed_dir / _frame_filename(input_stem, idx)).is_file()
            for idx in range(start, end)
        )
    else:
        cached_count = 0
    if cached_count:
        log.info(
            "Cache breakdown: %d/%d keyed frames cached (re-used), %d to infer.",
            cached_count, n_frames, n_frames - cached_count,
        )

    cap = cv2.VideoCapture(str(input_path))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    proc = _open_ffmpeg_rgba_sink(width, height, framerate, output_path, args)
    assert proc.stdin is not None
    try:
        t0 = time.monotonic()
        last_log = t0
        for i in range(n_frames):
            global_idx = start + i
            keyed_path = _frame_path(keyed_dir, input_stem, global_idx)

            ret, frame_bgr = cap.read()  # always advance to keep cap in sync
            if not ret:
                log.warning("Video ended early at frame %d (expected %d)", i, n_frames)
                break

            if keyed_path is not None and keyed_path.is_file():
                # Cache hit — discard the frame we just read and reuse keyed PNG
                rgba_u8 = _read_cached_keyed(keyed_path)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                alpha_hint = alpha_provider(global_idx, frame_rgb)
                result = _run_engine(
                    engine, frame_rgb, alpha_hint,
                    despill_strength=args.despill_strength,
                    despeckle_size=args.despeckle_size,
                    refiner_scale=args.refiner_scale,
                    input_is_linear=args.input_is_linear,
                    post_process_on_gpu=args.gpu_post_processing,
                    screen_channel=screen_channel,
                )
                rgba_u8 = _pack_rgba_u8(result["fg"], result["alpha"].squeeze(-1))
                if keyed_path is not None:
                    _write_cached_keyed(keyed_path, rgba_u8)

            proc.stdin.write(rgba_u8.tobytes())

            now = time.monotonic()
            if now - last_log >= 5.0:
                done = i + 1
                elapsed = now - t0
                eta = elapsed / done * (n_frames - done)
                log.info("frame %d/%d (%.1f fps, ETA %.0fs)",
                         done, n_frames, done / elapsed, eta)
                last_log = now
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = proc.wait()
        if rc != 0:
            sys.exit(f"ffmpeg exited with code {rc}")
        if handler is not None:
            handler.cleanup()
        del handler, engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clean up the tempdir holder for the no-cache GVM path.
        holder = getattr(args, "_tmp_alpha_holder", None)
        if holder is not None:
            holder.cleanup()

    log.info("Wrote %s", output_path)


# --- CLI ---------------------------------------------------------------------

def _resolve_output(input_path: Path, output: str | None, *, is_video: bool) -> Path:
    if output is not None:
        return Path(output)
    suffix = ".webm" if is_video else ".png"
    return input_path.with_name(input_path.stem + "_keyed" + suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input image (.png/.jpg/.webp) or video (.mp4/.mov/.mkv/.avi/.webm)")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output path (default: <stem>_keyed.png for image, <stem>_keyed.webm for video)",
    )

    # --- Engine / inference options (apply to both image and video) ----------
    common = parser.add_argument_group("inference")
    common.add_argument("--device", default=None,
                        help="Torch device: cuda, mps, or cpu (default: auto-detect)")
    common.add_argument("--backend", default=None, choices=("auto", "torch", "mlx"),
                        help="Inference backend (default: auto-detect). MLX = Apple Silicon only.")
    common.add_argument("--screen-color", default="auto",
                        choices=("auto", "green", "blue"),
                        help="Screen color to key against (default: auto). 'auto' probes "
                             "the first frame + its alpha hint and picks green vs blue "
                             "(matches the wizard's default).")
    common.add_argument("--despill-strength", type=float, default=0.5,
                        help="Despill strength 0.0-1.0 (default: 0.5, matches the wizard). "
                             "1.0 = full despill, 0.0 = no despill.")
    common.add_argument("--despeckle-size", type=int, default=400,
                        help="Min connected-pixel area for alpha matte cleanup (default: 400)")
    common.add_argument("--no-despeckle", action="store_true",
                        help="Disable matte cleanup. Equivalent to --despeckle-size 0.")
    common.add_argument("--refiner-scale", type=float, default=1.0,
                        help="Multiplier for refiner deltas (default: 1.0)")
    common.add_argument("--image-size", type=int, default=2048,
                        help="Inference image size used by the model (default: 2048)")
    common.add_argument("--input-is-linear", action="store_true",
                        help="Treat the input as linear (instead of sRGB)")
    common.add_argument("--gpu-post-processing", action="store_true",
                        help="Run resize / despeckle / despill / composite on the GPU "
                             "(torch path) instead of CPU (opencv/numpy). Default: off, "
                             "matching the wizard. The CPU path uses Lanczos4 for the "
                             "final resize, which is sharper at high output resolutions.")

    # --- Alpha hint generation ------------------------------------------------
    hint = parser.add_argument_group("alpha hint")
    hint.add_argument("--alpha-method", default="birefnet",
                      choices=("birefnet", "gvm", "videomama"),
                      help="Alpha hint generator for video input (default: birefnet). "
                           "'birefnet' is per-frame segmentation. "
                           "'gvm' runs the auto-matte path (wizard's `g` action): GVM "
                           "generates per-frame masks for the whole clip, then the CK "
                           "engine streams inference + encoding. "
                           "'videomama' refines a coarse mask hint provided via "
                           "--mask-hint (wizard's `v` action).")
    hint.add_argument("--birefnet-usage", default="Matting", metavar="USAGE",
                      help="BiRefNet model variant (default: Matting). See BiRefNetModule docs.")
    hint.add_argument("--alpha-hint", default=None,
                      help="Image-only: path to a pre-computed alpha hint PNG (bypasses BiRefNet)")
    hint.add_argument("--mask-hint", default=None, metavar="SOURCE",
                      help="VideoMaMa-only. Three ways to specify the coarse mask hint:\n"
                           "  birefnet|gvm  use that method's cached hints; if they aren't "
                           "cached yet for this input, run the method first to generate them.\n"
                           "  PATH          external directory of mask images (PNG/JPG/EXR) "
                           "or a video file — used as-is, no auto-generate.\n"
                           "  (omitted)     auto-detect cached hints (prefers alphahint_gvm/, "
                           "falls back to alphahint_birefnet/); errors out if neither exists.\n"
                           "All hints are force-thresholded to binary before VideoMaMa.")
    hint.add_argument("--videomama-chunk-size", type=int, default=50, metavar="N",
                      help="VideoMaMa-only: number of frames per inference chunk "
                           "(default: 50, matches the wizard).")

    # --- Video-only options ---------------------------------------------------
    video = parser.add_argument_group("video output (.webm)")
    video.add_argument("--start-frame", type=int, default=0,
                       help="Skip this many frames at the start of the video (default: 0)")
    video.add_argument("--end-frame", type=int, default=None,
                       help="Stop at this frame index (exclusive; default: end of video)")
    video.add_argument("--framerate", type=float, default=None,
                       help="Output framerate (default: input video's fps)")
    video.add_argument("--scale", default=None,
                       help='ffmpeg scale expression, e.g. "758:426" or "iw/2:-1"')
    video.add_argument("--crop", default=None,
                       help='ffmpeg crop expression "W:H:X:Y" (applied before scale)')
    video.add_argument("--crf", type=int, default=None,
                       help="VBR constant-quality 0-63 (lower=better). Disables lossless.")
    video.add_argument("--deadline", default="good", choices=("best", "good", "realtime"),
                       help="libvpx-vp9 deadline (default: good)")
    video.add_argument("--cpu-used", type=int, default=0,
                       help="libvpx-vp9 cpu-used (0=slowest/best quality, default: 0)")

    cache = parser.add_argument_group("cache (video only)")
    cache.add_argument("--cache-dir", default=None,
                       help="Parent dir for the shared cache (default: cwd). "
                            "Layout: <root>/corridorkey_cache/{alphahint_<method>,keyed}/"
                            "<input_stem>_NNNNNN.png — multiple inputs share one folder; "
                            "the input stem in each filename keeps them separated.")
    cache.add_argument("--no-cache", action="store_true",
                       help="Disable on-disk caching of alpha hints + keyed RGBA frames. "
                            "Every run does the full pipeline.")
    cache.add_argument("--clean", action="store_true",
                       help="Delete THIS input's cached files (matched by stem prefix) "
                            "before running. Other inputs in the shared cache are left "
                            "intact. Use this when you change inference params (despill, "
                            "refiner, image-size, screen-color) and want stale keyed "
                            "frames regenerated.")

    args = parser.parse_args()

    if args.no_despeckle:
        args.despeckle_size = 0

    if args.mask_hint is not None:
        if args.alpha_method != "videomama":
            log.warning("--mask-hint is only used with --alpha-method=videomama; ignoring.")
        elif args.mask_hint in ("birefnet", "gvm"):
            pass  # method-name path; existence is irrelevant
        elif not Path(args.mask_hint).exists():
            sys.exit(f"--mask-hint not found: {args.mask_hint}")

    args.device = resolve_device(args.device)
    log.info("Using device: %s", args.device)

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Input file not found: {args.input}")

    ext = input_path.suffix.lower()
    if ext in IMAGE_EXTS:
        is_video = False
    elif ext in VIDEO_EXTS:
        is_video = True
    else:
        sys.exit(f"Unsupported extension {ext!r}. "
                 f"Image: {sorted(IMAGE_EXTS)}; video: {sorted(VIDEO_EXTS)}")

    output_path = _resolve_output(input_path, args.output, is_video=is_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Routing: %s → %s (%s)", input_path, output_path, "video" if is_video else "image")

    # --clean: remove THIS input's cached files from the shared cache before
    # running (video only — image path doesn't cache).  Other inputs' caches
    # are left intact, since they share the same alphahint_*/ and keyed/ dirs.
    if args.clean and is_video:
        cache_root = _resolve_cache_root(args)
        if cache_root is None:
            log.info("--clean: no cache to remove (--no-cache is set)")
        else:
            removed = _clear_input_files(cache_root, input_path.stem)
            log.info("--clean: removed %d cached files for %r", removed, input_path.stem)

    if is_video:
        run_video(input_path, output_path, args)
    else:
        run_image(input_path, output_path, args)
    log.info("Done.")


if __name__ == "__main__":
    main()
