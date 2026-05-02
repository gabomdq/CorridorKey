# Changelog

## 2026-02-05 — 8GB GPU Memory + RGBA Comp Fixes

### GVM CUDA Out-of-Memory Fix (8GB cards)
- **`gvm_core/wrapper.py`**: Added `decode_chunk_size` parameter (default 1) to `process_sequence`, limiting VAE decode to one latent frame at a time. Auto-detects larger chunks for non-CUDA or high-VRAM devices.
- **`gvm_core/gvm/pipelines/pipeline_gvm.py`**: Modified `decode()` to accept `decode_chunk_size` and loop over latent frames in chunks, keeping peak VRAM at ~1-frame level.
- **`CorridorKey_DRAG_CLIPS_HERE_local.sh`**, **`RunGVMOnly.sh`**, **`RunInferenceOnly.sh`**: Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent CUDA memory fragmentation.

### Comp PNGs now include alpha channel (RGBA)
- **`clip_manager.py`** (primary output path): Comp PNGs now written as BGRA (4-channel) by merging the predicted alpha into the PNG's alpha channel, instead of 3-channel BGR.
- **`backend/service.py`**: `_write_outputs` updated to output BGRA PNGs with embedded alpha for the service API path.