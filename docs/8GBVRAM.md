# CorridorKey 8GB VRAM Compatibility Changes

## Problem

Running the GVM (General Video Matting) auto-matte pipeline on an 8GB GPU caused `torch.OutOfMemoryError` during the VAE temporal decoder's upsampling convolution.

**Error trace:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 688.00 MiB.
GPU 0 has a total capacity of 7.66 GiB of which 640.06 MiB is free.
Of the allocated memory 4.26 GiB is allocated by PyTorch, and 1.60 GiB is
reserved by PyTorch but unallocated.
```

**Error location:** `pipeline_gvm.py:85` → `vae.decode()` → `AutoencoderKLTemporalDecoder.forward()` → `UNet3DBlocks.forward()` → `Upsample2D.forward()` → `F.conv2d()`

## Root Cause Analysis

The crash occurs at the very end of the inference pipeline, after the UNet denoising is complete, during the final VAE decode step that converts latents back to image/alpha space.

1. **Hardcoded 1024p processing resolution**: The original code always resized input frames to a minimum height of 1024 pixels, regardless of available VRAM. At 1024p, the VAE decoder's intermediate feature maps in the upsampling path are large enough to require ~688 MiB per allocation, which exceeds the ~640 MiB free on an 8GB card after the UNet and other model components are loaded.

2. **VAE encode processed all frames at once**: The `encode()` method flattened all batch frames and ran the VAE encoder on the entire stack in a single call, allocating large contiguous tensors.

3. **No explicit memory cleanup between pipeline stages**: Intermediate tensors from the UNet inference (latents, noise tensors, image embeddings) were still resident when the VAE decode launched, fragmenting the available memory.

4. **PyTorch allocator fragmentation**: 1.60 GiB was "reserved but unallocated" — PyTorch had freed these blocks but the default caching allocator couldn't reuse them effectively for the 688 MiB allocation. The error message specifically recommended `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Changes Made

### 1. `gvm_core/wrapper.py` — Adaptive Resolution Scaling

Added `GVMProcessor._get_optimal_processing_resolution(orig_h, orig_w)` — a static method that queries the GPU's **total VRAM** via `torch.cuda.get_device_properties(0).total_memory` and selects an appropriate processing height:

| VRAM (GiB) | Processing Height | Max Long Edge |
|-----------|-------------------|---------------|
| ≥ 14      | 1024              | 1920          |
| ≥ 10      | 768               | 1536          |
| ≥ 6       | 576               | 1280          |
| < 6       | 512               | 1024          |

The method respects the original aspect ratio (recalculates `max_size` from `target_h`) and avoids upscaling frames that are already smaller than the target height.

A log message announces the detected VRAM and chosen resolution so users can confirm the heuristic.

The pipeline call now passes `encode_chunk_size=1` to process one frame at a time through the encoder.

### 2. `gvm_core/gvm/pipelines/pipeline_gvm.py` — Chunked VAE + Memory Cleanup

**`encode()` method:**
- Now accepts `encode_chunk_size` (default 8) and splits the input tensor into chunks
- Each chunk is encoded independently, and intermediate chunk tensors are explicitly deleted with `del`
- The chunk results are concatenated into a single latent tensor

**`decode()` method:**
- Added `torch.cuda.empty_cache()` after every chunk decode to return freed memory to the OS/allocator immediately
- Added explicit `del chunk` and `del latents` to release references promptly

**`__call__()` and `single_infer()`:**
- Added `encode_chunk_size` parameter throughout the call chain
- Added `torch.cuda.empty_cache()` right before the final decode operation to flush any lingering UNet intermediates
- All `single_infer()` invocations now pass `encode_chunk_size` through

### 3. `CorridorKey_DRAG_CLIPS_HERE_local.sh` — Launch Environment

Added `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching the Python process. This instructs PyTorch's CUDA caching allocator to use expandable memory segments, which dramatically reduces fragmentation by allowing the allocator to grow existing segments instead of requiring new contiguous blocks.

This matches the recommendation in the original error message:
> "If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation."

### Why `clip_manager.py` Was Unchanged

The existing call in `clip_manager.py` already uses the most conservative settings:
- `num_frames_per_batch=1` — processes one frame per batch
- `decode_chunk_size=1` — decodes one frame at a time

These are optimal for VRAM-constrained cards and required no modification.

## How Resolution Reduction Helps

VAE decoder memory usage scales quadratically with spatial resolution:

- **1024p**: ~688 MiB per upsampling conv allocation → OOM on 8GB
- **576p**: ~44% reduction in spatial dimensions → ~(576/1024)² ≈ 32% of peak allocation → ~220 MiB, fits comfortably

The VAE latent space is 1/8 of the pixel resolution, so at 1024p input the latent is 128×? pixels. At 576p it becomes 72×? pixels. Every subsequent upsampling block in the temporal decoder has proportionally smaller feature maps.

## Verification

To verify the changes work, run:
```bash
./CorridorKey_DRAG_CLIPS_HERE_local.sh /path/to/test/clip
```

You should see in the log output:
```
INFO Detected 7.7 GiB VRAM -> processing at 576p (max dimension XXXXpx)
```

The GVM auto-matte should complete without CUDA out-of-memory errors.
