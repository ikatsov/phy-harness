"""VLM-based video transcriber: sample every N-th frame and describe each frame."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.context import ValidationContext
from robot_manipulation_sim.validation.util import coerce_bool


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _sample_video_frames(
    video_path: Path,
    *,
    every_n_frames: int,
    max_frames: int,
) -> tuple[list[int], list[np.ndarray], float | None]:
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta["fps"]) if "fps" in meta and meta["fps"] else None
        idxs: list[int] = []
        frames: list[np.ndarray] = []
        for i, frame in enumerate(reader):
            if i % every_n_frames != 0:
                continue
            idxs.append(i)
            frames.append(np.asarray(frame))
            if len(frames) >= max_frames:
                break
        return idxs, frames, fps
    finally:
        reader.close()


def _build_gemini_model(model_name: str) -> Any:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pip install 'robot-manipulation-sim[vlm]' (google-generativeai, pillow)"
        ) from exc

    key = _api_key()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    genai.configure(api_key=key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.GenerationConfig(temperature=0.15),
    )


def _describe_frame_with_gemini(*, model: Any, prompt: str, frame_rgb: np.ndarray) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install 'robot-manipulation-sim[vlm]' (pillow)") from exc

    response = model.generate_content([prompt, Image.fromarray(frame_rgb)])
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("empty Gemini response")
    return text


class VlmVideoTranscriberAnalyzer:
    """Sample frames and emit a detailed JSON frame transcript."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def analyze(self, ctx: ValidationContext) -> AnalyzerResult:
        video = ctx.simulation.video
        if not video.is_file():
            return AnalyzerResult(
                "vlm_video_transcriber",
                ok=False,
                exit_code=2,
                messages=[f"video not found: {video}"],
            )

        every_n = max(1, int(self.params.get("every_n_frames", 25)))
        max_frames = max(1, int(self.params.get("max_frames", 120)))
        model = str(self.params.get("model") or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
        dry_run = coerce_bool(self.params.get("dry_run"), default=False)
        out_raw = self.params.get("json_out")
        out_path = Path(out_raw).resolve() if out_raw else video.with_suffix(".vlm_transcript.json")

        frame_prompt = str(
            self.params.get(
                "frame_prompt",
                "Describe this simulation frame in detail: robot arm pose, gripper opening/contacts, object positions, and any visible motion cues.",
            )
        )
        if ctx.task_spec:
            frame_prompt = f"{frame_prompt}\nTask context: {ctx.task_spec.strip()}"

        try:
            idxs, frames, fps = _sample_video_frames(video, every_n_frames=every_n, max_frames=max_frames)
            if not frames:
                raise RuntimeError("no sampled frames")
            gemini_model = None if dry_run else _build_gemini_model(model)
            entries: list[dict[str, Any]] = []
            for i, frame in zip(idxs, frames):
                if dry_run:
                    desc = "dry-run: VLM call skipped"
                else:
                    desc = _describe_frame_with_gemini(model=gemini_model, prompt=frame_prompt, frame_rgb=frame)
                t_sec = (float(i) / fps) if fps and fps > 0 else None
                entries.append(
                    {
                        "frame_index": int(i),
                        "time_sec": t_sec,
                        "description": desc,
                    }
                )
            payload = {
                "analyzer": "vlm_video_transcriber",
                "video": str(video.resolve()),
                "every_n_frames": every_n,
                "sampled_frames": len(entries),
                "fps": fps,
                "entries": entries,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return AnalyzerResult(
                "vlm_video_transcriber",
                ok=True,
                exit_code=0,
                messages=[f"wrote {out_path.resolve()}"],
                artifacts={"json_out": str(out_path.resolve()), "entries": len(entries)},
            )
        except Exception as exc:  # noqa: BLE001
            return AnalyzerResult(
                "vlm_video_transcriber",
                ok=False,
                exit_code=2,
                messages=[f"{type(exc).__name__}: {exc}"],
            )
