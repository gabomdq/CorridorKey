@echo off
TITLE VideoMaMa Setup Wizard
echo ===================================================
echo   VideoMaMa (AlphaHint Generator) - Auto-Installer
echo ===================================================
echo.

if not exist ".venv" (
    echo [ERROR] Project environment not found.
    echo Please run Install_CorridorKey_Windows.bat first!
    pause
    exit /b
)

set CHECKPOINTS_DIR=VideoMaMaInferenceModule\checkpoints
set VIDEOMAMA_DIR=%CHECKPOINTS_DIR%\VideoMaMa
set SVD_DIR=%CHECKPOINTS_DIR%\stable-video-diffusion-img2vid-xt

if not exist "%CHECKPOINTS_DIR%" mkdir "%CHECKPOINTS_DIR%"

:: --- 0. Migrate misinstalled layout from earlier installer versions --------
:: Older installer dropped SammyLim/VideoMaMa contents directly into
:: checkpoints/, but inference.py expects them under VideoMaMa/.
if exist "%CHECKPOINTS_DIR%\unet" (
    if not exist "%VIDEOMAMA_DIR%\unet" (
        echo [migrate] Moving misinstalled VideoMaMa files into %VIDEOMAMA_DIR% ...
        if not exist "%VIDEOMAMA_DIR%" mkdir "%VIDEOMAMA_DIR%"
        move "%CHECKPOINTS_DIR%\unet" "%VIDEOMAMA_DIR%\unet"
        if exist "%CHECKPOINTS_DIR%\dino_projection_mlp.pth" move "%CHECKPOINTS_DIR%\dino_projection_mlp.pth" "%VIDEOMAMA_DIR%\"
        if exist "%CHECKPOINTS_DIR%\.gitattributes" move "%CHECKPOINTS_DIR%\.gitattributes" "%VIDEOMAMA_DIR%\"
        if exist "%CHECKPOINTS_DIR%\README.md" move "%CHECKPOINTS_DIR%\README.md" "%VIDEOMAMA_DIR%\"
        echo [migrate] Done.
    )
)

:: --- 1. SammyLim/VideoMaMa (UNet + DINO projection) ------------------------
if exist "%VIDEOMAMA_DIR%\unet\diffusion_pytorch_model.safetensors" (
    echo [1/2] VideoMaMa weights already present at %VIDEOMAMA_DIR% - skipping.
) else (
    echo [1/2] Downloading VideoMaMa weights from HuggingFace ...
    uv run hf download SammyLim/VideoMaMa --local-dir %VIDEOMAMA_DIR%
)

:: --- 2. Stable Video Diffusion base model ----------------------------------
if exist "%SVD_DIR%\model_index.json" (
    echo [2/2] SVD base model already present at %SVD_DIR% - skipping.
) else (
    echo [2/2] Downloading Stable Video Diffusion base model ^(~10 GB^) ...
    echo       This is gated on HuggingFace.  If the download fails with
    echo       a 401 error, run:
    echo           uv run hf auth login
    echo       and accept the license at:
    echo           https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
    echo       then re-run this installer.
    uv run hf download stabilityai/stable-video-diffusion-img2vid-xt --local-dir %SVD_DIR%
    if errorlevel 1 (
        echo.
        echo [ERROR] SVD base download failed.  See the instructions above
        echo         and re-run this installer.
        pause
        exit /b
    )
)

echo.
echo ===================================================
echo   VideoMaMa Setup Complete!
echo ===================================================
echo   %VIDEOMAMA_DIR%
echo   %SVD_DIR%
pause
