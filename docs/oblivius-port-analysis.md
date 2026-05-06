# Oblivius perf/fix port analysis

Survey of every "perf:" and "fix:" commit on the `99oblivius/CorridorKey-Engine`
fork (range `origin/main..oblivius/main`, ~70 commits) for portability onto
this repo's `main`.  Each row records the intent, what main looks like in
the same neighbourhood, and the decision: **port / skip / N/A**.

> **Architectural blocker.** Oblivius split `CorridorKeyModule/inference_engine.py`
> into `base_engine.py` (1518 lines), `inference_engine.py` (59), `optimized_engine.py`
> (106), `optimization_config.py` (297), `engine_factory.py`, and `constants.py`.
> Many oblivius commits modify `base_engine.py`, the new `ck_engine/` package, or
> infrastructure (`pp_stream`, drain pool, process-per-GPU) that doesn't exist on
> main.  A literal cherry-pick fails — analysis below is **intent-based**.

## Decision matrix

| Commit | Intent (one line) | Main analog | Decision |
|---|---|---|---|
| `f2f1ffb` (fix) | fp16 dtype errors in compiled postprocess (gaussian_blur, lerp) | [color_utils.py:503](../CorridorKeyModule/core/color_utils.py#L503) `clean_matte_torch` calls `TF.gaussian_blur` directly | **Port partial** — wrap blur with fp32 cast; checkerboard-lerp portion is in `base_engine.py`-only code, N/A |
| `b92e344` (fix) | clone model outputs before pp_stream to dodge CUDA-graph buffer reuse | main has no `pp_stream`; inference + postprocess on the same stream | **Skip** — no pp_stream means no cross-stream race |
| `8910389` (fix) | pyexr writer channel scrambling on non-contiguous slices | main uses `cv2.imwrite`; no pyexr writer | **N/A** |
| `b604ded` (fix) | piecewise sRGB OETF for EXR sequences | [color_utils.py:55-72](../CorridorKeyModule/core/color_utils.py#L55) — already piecewise sRGB | **N/A — already correct** |
| `65de2c5` (fix) | torch.compile reduce-overhead/max-autotune CUDA-graph failures (5 sub-fixes) | Main uses `mode="max-autotune"` ([inference_engine.py:185](../CorridorKeyModule/inference_engine.py#L185)) and is exposed to the same class of bug | **Port 2/6** (see below) |
| `df79b15` (fix) | `total_mem` → `total_memory` attribute name | [device_utils.py:261](../device_utils.py#L261) already `total_memory` | **N/A — already correct** |
| `9ba08af` (perf) | BCHW GPU postprocessing + connected-components matte cleanup | main's `_postprocess_torch` already runs in BCHW until the final `permute(0,2,3,1)` | **Skip — already BCHW** |
| `61e7153` (perf) | CHW zero-copy EXR + pp_stream overlap + O(1) stats + lock defaults | All four require `pp_stream`/`gpu_worker.py` infrastructure that doesn't exist on main | **Skip — infrastructure-bound** |
| `cb4929c` (perf) | direct-pinned EXR writes via `PendingTransfer._direct_views` | `PendingTransfer` doesn't exist on main; main writes via cv2 in `clip_manager.py` | **N/A** |
| `9a1b919` (perf) | compile postprocess + sync-free matte cleanup + drain buffer pool + several utility-level wins | Drain pool and compile-of-postprocess are infra-bound. **Several `core/color_utils.py` fragments are clean wins on main** | **Port partial** — see below |
| `d181213` (perf) | pyexr fp16 EXR writer + cv2 4.13 DWAB corruption workaround | Main writes PXR24 (not DWAB), so the cv2 corruption bug doesn't bite. Adding pyexr is a new dep + new code path | **Skip** — main's PXR24 path is unaffected by the documented cv2 4.13 DWAB/DWAA bug |
| `3826e91` (perf) | GPU color_utils — `despill_torch`, `connected_components`, `clean_matte_torch`, `checkerboard` | Already on main via `aa3c9a9` ([core/color_utils.py](../CorridorKeyModule/core/color_utils.py)) | **N/A — already on main (#172)** |
| `65fdd01` (perf) | GPU preprocessing + batching for `process_frame` | Already on main via `aa3c9a9` ([inference_engine.py:216](../CorridorKeyModule/inference_engine.py#L216)) | **N/A — already on main (#172)** |
| `de27285` (perf) | pinned memory staging for H2D uploads in inference hot path | Main does `torch.from_numpy(...).to(device, non_blocking=True)` — non_blocking is a no-op without pinned source | **Port** — caches a pinned float32 buffer per-shape on the engine and stages through it |
| `64c8943` (perf) | enable GPU pre/postprocessing on MPS devices | Main gates GPU postprocess on `post_process_on_gpu` flag (caller-controlled), no MPS-specific block. Calling code at clip_manager would have to honour MPS | **Skip** — MPS users on main already pass `post_process_on_gpu=True` and it works; oblivius's gate is on `use_cuda` only, which main doesn't have. No bug to fix here. |
| `0e140e2` (perf) | iterated small-kernel max_pool2d for GPU matte dilation + gaussian_blur | [color_utils.py:494-498](../CorridorKeyModule/core/color_utils.py#L494) already iterates `F.max_pool2d` with kernel 5 | **N/A — already on main** |
| `ed5662d` (feat) | process-per-GPU pipeline + output layer selection + DWAB EXR | Process-per-GPU N/A; output layer selection requires settings infra | **Skip — infrastructure-bound** |
| `e2eaf7a` (refactor) | prep for process-per-GPU | Same — N/A | **N/A** |

## What gets ported (in order of confidence)

1. **`65de2c5#1` Refiner scale: Python float → device tensor.**  
   Main currently registers/unregisters a `forward_hook` on `self.model.refiner`
   per call ([inference_engine.py:464-470, 478-479](../CorridorKeyModule/inference_engine.py#L464)),
   capturing `refiner_scale` (a Python float) in the closure.  Under
   `torch.compile(mode="max-autotune")` (which main enables on CUDA), the float
   gets baked into the captured graph on the first call, so subsequent calls
   with a different `refiner_scale` silently use the cached value.  The fix:
   register the hook **once** at engine init (before `_compile`), make it
   read from a 1-element tensor on device, and update via `.fill_()`.  Hook
   always multiplies — no Python branch.

2. **`65de2c5#6` `out += residual` → `out = out + residual` in `RefinerBlock`.**
   In-place tensor mutation can corrupt under CUDA graphs (aliased buffers);
   one-line change in [model_transformer.py:94](../CorridorKeyModule/core/model_transformer.py#L94).

3. **`de27285` Pinned-memory H2D staging.**  Cache a pinned float32 buffer per
   input shape on the engine, copy `numpy → pinned`, then `pinned → device`
   with `non_blocking=True` so the DMA is genuinely async on CUDA.  Falls
   back to the existing path on non-CUDA devices.  Cost: one extra CPU copy
   per frame, gated by shape change.  Benefit: H2D can overlap with the
   previous frame's compute tail.

4. **`9a1b919` `core/color_utils.py` GPU-side fusions.**  Three independent wins:
   - **`_linear_to_srgb_torch` / `_srgb_to_linear_torch`**: tensor-only fast
     paths that use `clamp(min=...)` directly and avoid building an explicit
     mask tensor.  Fewer temporaries, more compile-friendly.
   - **`composite_straight` → `torch.lerp(bg, fg, alpha)`** / **`composite_premul` → `torch.addcmul(fg, bg, 1-alpha)`** when
     inputs are tensors.  Fused ops, fewer intermediate allocations.
   - **In-place `despill_torch`**: replace `torch.stack` of three new tensors
     with in-place ops on a clone of the input.  Preserves main's
     `screen_channel` parameter (oblivius dropped it; main keeps blue-screen
     support from PR #241).

5. **`f2f1ffb` fp16-safe gaussian_blur in `clean_matte_torch`.**
   `TF.gaussian_blur` builds the kernel via `torch.linspace`, which Dynamo
   can't trace in fp16.  Cast input → fp32 → blur → cast back.  Three-line
   change.  Defense-in-depth: main's `clean_matte_torch` is currently not
   wrapped in `torch.compile`, but the model's `model_dtype` can be fp16
   and the same alpha tensor flows through, so this is a good guardrail
   that costs nothing at runtime in the fp32 case.

## What gets skipped (and why)

- **Anything that needs `pp_stream`**: cross-stream cloning, sync-free matte
  cleanup, compile of postprocess, BCHW pp_stream overlap.  Adding
  `pp_stream` correctly is a multi-day rewrite of the I/O pipeline and
  changes the contract between `process_frame` and the `_save_*` callers
  in `clip_manager.py`.  Not worth the disruption for the wins observed.

- **Anything that needs `PendingTransfer` / `gpu_worker`**: direct-pinned
  drain, drain pool, output-layer selection.  Same reason — these are the
  drain-pool architecture that doesn't exist on main.

- **Anything that needs the `OptimizationConfig` dataclass**: cache-clearing
  config propagation, tiled-refiner auto-swap, comp_format/comp_checkerboard
  toggles.  These hang off oblivius's settings dataclass; main hard-codes
  the equivalent behaviour.  Sub-fixes 2-5 of `65de2c5` fall here.

- **Anything claiming "(upstream aa3c9a9)" or "(upstream #172)"**: that work
  is already on main.  Re-applying would conflict.

## Open follow-ups

- **EXR writer**: main writes EXR fp16 via `cv2.IMWRITE_EXR_TYPE_HALF`
  ([backend/frame_io.py:27-32](../backend/frame_io.py#L27)).  Oblivius
  documents that cv2 4.13.0 corrupts when both `IMWRITE_EXR_TYPE` and
  DWAB/DWAA are set.  Main avoids DWAB so probably fine, but worth a
  one-time round-trip test on the user's actual cv2 version to be certain.

- **`pp_stream` adoption**: if benchmarking shows main is overlap-bound
  on a specific workload (large clips, slow disk), revisit.  The wins
  oblivius reports (200-300 ms/frame on 4K) are real but require taking
  on the drain-pool architecture wholesale.

- **pyexr backend**: opt-in EXR writer for fp16-native output and DWA
  compression.  Adds an optional dep.  Worth doing if EXR output is on
  the critical path; unclear from the rancho workload (which writes PNG).
