#!/usr/bin/env bash

cd "$(dirname "$0")"

# Set the Terminal window title
echo -n -e "\033]0;VideoMaMa Setup Wizard\007"
echo "==================================================="
echo "   VideoMaMa (AlphaHint Generator) - Auto-Installer"
echo "==================================================="
echo ""

if [ ! -d ".venv" ]; then
    echo "[ERROR] Project environment not found."
    echo "Please run Install_CorridorKey_Linux_Mac.sh first!"
    read -p "Press [Enter] to exit..."
    exit 1
fi

CHECKPOINTS_DIR="VideoMaMaInferenceModule/checkpoints"
VIDEOMAMA_DIR="$CHECKPOINTS_DIR/VideoMaMa"
SVD_DIR="$CHECKPOINTS_DIR/stable-video-diffusion-img2vid-xt"

mkdir -p "$CHECKPOINTS_DIR"

# --- 0. Migrate misinstalled layout from earlier installer versions -----------
# The previous installer downloaded SammyLim/VideoMaMa directly into
# checkpoints/ (so the unet/ and dino_projection_mlp.pth ended up loose
# next to the .cache/), but inference.py expects them under VideoMaMa/.
# Move them into place if we detect the old layout.
if [ -d "$CHECKPOINTS_DIR/unet" ] && [ ! -d "$VIDEOMAMA_DIR/unet" ]; then
    echo "[migrate] Moving misinstalled VideoMaMa files into $VIDEOMAMA_DIR ..."
    mkdir -p "$VIDEOMAMA_DIR"
    mv "$CHECKPOINTS_DIR/unet" "$VIDEOMAMA_DIR/unet"
    if [ -f "$CHECKPOINTS_DIR/dino_projection_mlp.pth" ]; then
        mv "$CHECKPOINTS_DIR/dino_projection_mlp.pth" "$VIDEOMAMA_DIR/"
    fi
    if [ -f "$CHECKPOINTS_DIR/.gitattributes" ]; then
        mv "$CHECKPOINTS_DIR/.gitattributes" "$VIDEOMAMA_DIR/"
    fi
    if [ -f "$CHECKPOINTS_DIR/README.md" ]; then
        mv "$CHECKPOINTS_DIR/README.md" "$VIDEOMAMA_DIR/"
    fi
    if [ -d "$CHECKPOINTS_DIR/.cache" ]; then
        # Leave the HF download cache alone — it's harmless once content is moved.
        :
    fi
    echo "[migrate] Done."
fi

# --- 1. SammyLim/VideoMaMa (UNet + DINO projection) --------------------------
if [ -d "$VIDEOMAMA_DIR/unet" ] && [ -f "$VIDEOMAMA_DIR/unet/diffusion_pytorch_model.safetensors" ]; then
    echo "[1/2] VideoMaMa weights already present at $VIDEOMAMA_DIR — skipping."
else
    echo "[1/2] Downloading VideoMaMa weights from HuggingFace ..."
    uv run hf download SammyLim/VideoMaMa --local-dir "$VIDEOMAMA_DIR"
fi

# --- 2. Stable Video Diffusion base model ------------------------------------
# inference.py loads feature_extractor / image_encoder / vae from this path,
# so VideoMaMa cannot run without it.  This is the upstream Stability AI
# repo (gated on HuggingFace — accept the license + login first if you
# haven't already).
if [ -f "$SVD_DIR/model_index.json" ]; then
    echo "[2/2] SVD base model already present at $SVD_DIR — skipping."
else
    echo "[2/2] Downloading Stable Video Diffusion base model (~10 GB) ..."
    echo "      This is gated on HuggingFace.  If the download fails with"
    echo "      a 401 error, run:"
    echo "          uv run hf auth login"
    echo "      and accept the license at:"
    echo "          https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt"
    echo "      then re-run this installer."
    uv run hf download stabilityai/stable-video-diffusion-img2vid-xt \
        --local-dir "$SVD_DIR" \
        || {
            echo ""
            echo "[ERROR] SVD base download failed.  See the instructions above"
            echo "        and re-run this installer."
            read -p "Press [Enter] to close..."
            exit 1
        }
fi

echo ""
echo "==================================================="
echo "  VideoMaMa Setup Complete!"
echo "==================================================="
echo "  $VIDEOMAMA_DIR"
echo "  $SVD_DIR"
read -p "Press [Enter] to close..."
