#!/usr/bin/env bash

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

# Set PyTorch CUDA memory allocation strategy to avoid fragmentation
# This helps on 8GB and similar VRAM-constrained cards
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting CorridorKey Inference..."
echo "Scanning ClipsForInference for Ready Clips (Input + Alpha)..."

# Run via uv entry point (handles the virtual environment automatically)
uv run corridorkey run-inference

echo "Inference Complete."
