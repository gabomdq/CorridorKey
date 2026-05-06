"""Encode a directory of RGBA PNG frames into a transparent VP9 WebM.

Designed for CorridorKey composed output (e.g. ``<project>/Output/Comp/NNNNN.png``)
but works on any zero-padded numeric PNG sequence. Wraps the standard 2-pass
``libvpx-vp9`` recipe so the manual ``ffmpeg`` invocation does not need to be
re-typed for every clip.

Usage:
    uv run python scripts/compose_pngs_to_webm.py \\
        ~/thegaucho/ai/rancho/chanchos1/Output/Comp \\
        chanchos1.webm \\
        --framerate 10 --scale 758:426 --skip-start 2 --skip-end 5
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)\.png$", re.IGNORECASE)


def _discover_sequence(directory: Path) -> tuple[str, int, int, int]:
    """Return (prefix, pad_width, first_index, last_index) for the PNG sequence.

    The sequence's prefix and zero-pad width are inferred from the most
    common pattern in the directory, so siblings like ``thumb.png`` are
    ignored without complaint.
    """
    buckets: dict[tuple[str, int], list[int]] = {}
    for entry in directory.iterdir():
        match = _FRAME_RE.match(entry.name)
        if not match:
            continue
        key = (match.group("prefix"), len(match.group("num")))
        buckets.setdefault(key, []).append(int(match.group("num")))

    if not buckets:
        sys.exit(f"No NNNN.png frames found in {directory}")

    (prefix, pad), nums = max(buckets.items(), key=lambda kv: len(kv[1]))
    nums.sort()
    return prefix, pad, nums[0], nums[-1]


def _build_vf(args: argparse.Namespace) -> str | None:
    chain: list[str] = []
    if args.crop:
        chain.append(f"crop={args.crop}")
    if args.scale:
        chain.append(f"scale={args.scale}")
    return ",".join(chain) if chain else None


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="directory of NNNNN.png frames")
    parser.add_argument("output", type=Path, help="output .webm file")
    parser.add_argument("--framerate", type=float, default=30.0)
    parser.add_argument(
        "--scale",
        help='ffmpeg scale expression, e.g. "758:426" or "iw/2:-1"',
    )
    parser.add_argument(
        "--crop",
        help='ffmpeg crop expression "W:H:X:Y" (applied before scale)',
    )
    parser.add_argument(
        "--skip-start", type=int, default=0,
        help="drop this many frames from the beginning of the sequence",
    )
    parser.add_argument(
        "--skip-end", type=int, default=0,
        help="drop this many frames from the end of the sequence",
    )

    quality = parser.add_mutually_exclusive_group()
    quality.add_argument(
        "--lossless", dest="lossless", action="store_true", default=True,
        help="encode with -lossless 1 (default)",
    )
    quality.add_argument(
        "--crf", type=int,
        help="constant-quality VBR (0-63, lower=better); disables lossless",
    )

    parser.add_argument("--deadline", default="good", choices=["best", "good", "realtime"])
    parser.add_argument("--cpu-used", type=int, default=0)
    parser.add_argument(
        "--two-pass", action="store_true",
        help="run 2-pass VBR analysis (off by default; meaningless for lossless)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found in PATH")
    if not args.input.is_dir():
        sys.exit(f"Not a directory: {args.input}")

    prefix, pad, first, last = _discover_sequence(args.input)
    start_number = first + args.skip_start
    end_number = last - args.skip_end
    if end_number < start_number:
        sys.exit("skip-start + skip-end leaves no frames to encode")
    n_frames = end_number - start_number + 1

    pattern = args.input / f"{prefix}%0{pad}d.png"
    vf = _build_vf(args)

    input_args = [
        "-framerate", str(args.framerate),
        "-start_number", str(start_number),
        "-i", str(pattern),
        "-frames:v", str(n_frames),
    ]

    enc = [
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-deadline", args.deadline,
        "-cpu-used", str(args.cpu_used),
    ]
    if args.crf is None:
        enc += ["-lossless", "1"]
    else:
        enc += ["-b:v", "0", "-crf", str(args.crf)]

    vf_args = ["-vf", vf] if vf else []

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.two_pass:
        _run(
            ["ffmpeg", "-y", *input_args, *vf_args, *enc, "-an", str(args.output)],
            dry_run=args.dry_run,
        )
        return

    with tempfile.TemporaryDirectory() as td:
        passlog = str(Path(td) / "ffmpeg2pass")
        pass1 = [
            "ffmpeg", "-y", *input_args, *vf_args, *enc,
            "-pass", "1", "-passlogfile", passlog,
            "-an", "-f", "null", "/dev/null",
        ]
        pass2 = [
            "ffmpeg", "-y", *input_args, *vf_args, *enc,
            "-pass", "2", "-passlogfile", passlog,
            "-an", str(args.output),
        ]
        _run(pass1, dry_run=args.dry_run)
        _run(pass2, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
