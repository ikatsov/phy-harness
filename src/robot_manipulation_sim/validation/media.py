"""FFmpeg-backed media helpers for rollout analyzers (clip, probe, frame extract)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def probe_duration_seconds(video: Path) -> float:
    r = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
        check=False,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr or "")
    if not m:
        return 8.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def clip_video_head(src: Path, dst: Path, max_seconds: float) -> None:
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-t",
            str(max_seconds),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )


def prepare_eval_video(video: Path, tmp: Path, max_duration_seconds: float) -> tuple[Path, float, float]:
    full = probe_duration_seconds(video)
    if full <= max_duration_seconds + 1e-3:
        return video, full, full
    out = tmp / "eval_clip.mp4"
    clip_video_head(video, out, max_duration_seconds)
    return out, full, min(full, max_duration_seconds)


def extract_frames(video: Path, out_dir: Path, *, max_frames: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f*.png"):
        old.unlink()
    dur = max(probe_duration_seconds(video), 0.25)
    fps = max(0.12, min(2.5, max_frames / dur))
    pattern = str(out_dir / "f%04d.png")
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(max_frames),
            pattern,
        ],
        check=True,
    )
    return sorted(out_dir.glob("f*.png"))
