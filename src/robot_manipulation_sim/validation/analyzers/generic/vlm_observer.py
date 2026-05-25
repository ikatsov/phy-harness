"""Gemini VLM observer analyzer (inline video or PNG frames)."""

from __future__ import annotations

import json
import math
import mimetypes
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.context import ValidationContext
from robot_manipulation_sim.validation.media import extract_frames, prepare_eval_video, probe_duration_seconds
from robot_manipulation_sim.validation.util import coerce_bool

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"(?s).*All support for the `google\.generativeai` package has ended.*",
)

OBSERVER_NEUTRAL_BLOCK = """
You are an expert observer of **simulated robot manipulation** recordings.

**Neutral stream (task-agnostic):** For **`summary_task_agnostic`** and **`second_by_second_neutral`**, describe
**only what pixels and any appended simulator metrics show**—joint motion, contacts, gripper, box motion, and what
each **2×2** tile shows. **Do not** import goals beyond what is visible; treat this stream as a **pure observer**
accounting of motion and scene dynamics.
""".strip()

TASK_EVAL_BLOCK = """
### Task-conditioned stream (required)
You MUST also output **`summary_task_evaluation`** and **`second_by_second_task`**, the same **temporal length**
as **`second_by_second_neutral`** (same number of entries; index **i** covers **[``i``, ``i`` + 1)** seconds from the
start of the evaluated clip).

- **`summary_task_evaluation`**: **2–5 sentences** judging how well the **observed behavior** matches the
  **User task specification** below (success, partial progress, wrong or ambiguous behavior, or **cannot tell**).
  Stay grounded in what you saw; cite conflicts with metrics if relevant.
- **`second_by_second_task`**: For each second **i**, one string (**≤ ~45 words**) describing **progress toward or
  deviation from** that user task during that second (e.g. "on task: only base rotation visible", "off task:
  clear elbow flex toward the box"). If the specification is empty or says there is no task, each entry must be
  exactly **`N/A (no task spec).`** and **`summary_task_evaluation`** must be exactly:
  **`No task specification was provided; task-aligned evaluation is not applicable.`**

**User task specification** (may be empty):
---
{task_spec_block}
---
""".strip()

MOTION_RUBRIC = """
### Temporal narration — neutral stream (required)
Provide **`second_by_second_neutral`**: an array of strings, one per second from the **start of the evaluated clip**.

- Each string: **which joints or links move**, direction when obvious, **gripper**, **base** vs arm, **box** / contacts,
  and **which 2×2 panel** helps if useful. **≤ ~45 words** per entry.

### Motion quality (watch the **entire** clip temporally — not a single frame)
- **Controlled**: joints move in a **coordinated, purposeful** way (e.g. slow base spin, steady reach-to-grasp). Velocities look bounded; one primary motion intent at a time is OK.
- **Chaotic / bad control** (set **`motion_controlled`: false** and **`pass`: false**): multiple links snapping or oscillating together, violent flailing, obvious high-frequency jitter/vibration, repeated slamming between extremes, or motion that looks **random / unstable** rather than deliberate.
- **Bias conservative**: if unsure whether motion is controlled, set **`motion_controlled`** to **false**.
- Do **not** praise "smooth" motion unless it truly looks smooth over time in all four camera panels.

### JSON (exact keys, no markdown)
{"pass": <bool>, "summary_task_agnostic": <string>, "summary_task_evaluation": <string>, "second_by_second_neutral": [<string>], "second_by_second_task": [<string>], "issues": [<string>], "panels_ok": <bool>, "motion_controlled": <bool>}

Rules:
1. **`motion_controlled`** must be **true** only for clearly controlled arm motion.
2. If **`motion_controlled`** is **false**, **`pass`** MUST be **false** (even if panels look fine).
3. **`pass`** is **true** only if **`motion_controlled`** AND **`panels_ok`** AND there is **no visible catastrophic collision** (self-collision, violent environment impact, or box ejected in an uncontrolled way). **`pass`** encodes **recording quality + safe-looking motion**, not “task succeeded.”
4. **`panels_ok`**: four tiles in a 2×2 grid are discernible (top row: **perspective RGB** ``overview`` | **wrist RGB**; bottom row: **matching grayscale depth** for those two views), not mostly black.
5. **`summary_task_agnostic`**: short **overall** neutral description (motion + contacts + panels), **without** duplicating full per-second neutral text.
6. **`summary_task_evaluation`**: per the **Task-conditioned stream** section above.
7. **`second_by_second_neutral`** and **`second_by_second_task`**: required; same length; follow **Clip timing** if present.
8. **`issues`**: include e.g. `"chaotic or jerky arm motion"` when applicable; else [].
"""


def build_prompt(
    *,
    metrics_text: str | None,
    media_hint: str,
    eval_duration_seconds: float | None = None,
    task_spec: str | None = None,
) -> str:
    clip_timing = ""
    if eval_duration_seconds is not None:
        dur = max(float(eval_duration_seconds), 0.05)
        n = max(1, math.ceil(dur))
        clip_timing = (
            f"**Clip timing (strict)**: The evaluated clip is **{dur:.3f} s** long. "
            f"Your **`second_by_second_neutral`** and **`second_by_second_task`** arrays MUST each contain "
            f"**exactly {n} strings** (indices **0** through **{n - 1}**), one per second as defined below.\n\n"
        )

    ts = (task_spec or "").strip()
    task_spec_block = ts if ts else "(none — use the N/A rules in the Task-conditioned stream section.)"

    body = (
        f"{OBSERVER_NEUTRAL_BLOCK}\n\n"
        f"{TASK_EVAL_BLOCK.format(task_spec_block=task_spec_block)}\n\n"
        f"{media_hint}\n\n"
        "The recording is ONE 2×2 grid: top-left = scene perspective RGB (``overview``), top-right = wrist RGB, "
        "bottom-left = grayscale depth for the perspective view, bottom-right = grayscale depth for the wrist view.\n\n"
        f"{clip_timing}"
        f"{MOTION_RUBRIC.strip()}\n"
    )
    if metrics_text:
        body += "\nSimulator metrics (pixels may disagree; mention in summaries or per-second lines if so):\n" + metrics_text.strip() + "\n"
    return body


def finalize_verdict(verdict: dict, *, expected_timeline_entries: int | None = None) -> dict:
    """Normalize keys, migrate legacy `summary` / `second_by_second`, optionally enforce timeline length."""
    out = dict(verdict)

    # Legacy single-stream keys → neutral stream
    if not isinstance(out.get("summary_task_agnostic"), str) or not str(out.get("summary_task_agnostic", "")).strip():
        if isinstance(out.get("summary"), str) and out["summary"].strip():
            out["summary_task_agnostic"] = out["summary"]
    if not isinstance(out.get("second_by_second_neutral"), list) or not out["second_by_second_neutral"]:
        legacy = out.get("second_by_second")
        if isinstance(legacy, list) and legacy:
            out["second_by_second_neutral"] = [str(x) for x in legacy]

    def _str_list(key: str) -> list[str]:
        raw = out.get(key)
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if raw is None:
            return []
        return [str(raw)]

    out["second_by_second_neutral"] = _str_list("second_by_second_neutral")
    out["second_by_second_task"] = _str_list("second_by_second_task")

    if not isinstance(out.get("summary_task_agnostic"), str):
        out["summary_task_agnostic"] = "" if out.get("summary_task_agnostic") is None else str(out["summary_task_agnostic"])
    if not isinstance(out.get("summary_task_evaluation"), str):
        out["summary_task_evaluation"] = (
            "" if out.get("summary_task_evaluation") is None else str(out["summary_task_evaluation"])
        )

    n_neutral = len(out["second_by_second_neutral"])
    n_task = len(out["second_by_second_task"])
    if n_neutral != n_task:
        issues = list(out.get("issues") or [])
        issues.append(
            f"VLM JSON: second_by_second_neutral len={n_neutral} vs second_by_second_task len={n_task}; padded/fixed"
        )
        out["issues"] = issues
        if n_neutral > n_task:
            out["second_by_second_task"].extend(["N/A (length mismatch)."] * (n_neutral - n_task))
        else:
            out["second_by_second_neutral"].extend(["N/A (length mismatch)."] * (n_task - n_neutral))

    if expected_timeline_entries is not None:
        exp = int(expected_timeline_entries)
        if exp < 1:
            exp = 1
        for key in ("second_by_second_neutral", "second_by_second_task"):
            arr = out[key]
            if len(arr) != exp:
                issues = list(out.get("issues") or [])
                issues.append(f"VLM JSON: expected {exp} timeline entries for {key}, got {len(arr)}")
                out["issues"] = issues
                out["pass"] = False
                if len(arr) < exp:
                    pad = "N/A (missing second)." if key == "second_by_second_neutral" else "N/A (missing second)."
                    arr = arr + [pad] * (exp - len(arr))
                else:
                    arr = arr[:exp]
                out[key] = arr

    if "motion_controlled" not in out:
        issues = list(out.get("issues") or [])
        issues.append("VLM JSON omitted motion_controlled; failing conservative")
        out["issues"] = issues
        out["motion_controlled"] = False
        out["pass"] = False
    elif out.get("motion_controlled") is False:
        out["pass"] = False
    elif out.get("panels_ok") is False:
        out["pass"] = False
    return out


def _gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _call_gemini_video_inline_google_genai(
    *,
    video_path: Path,
    model: str,
    prompt: str,
    expected_timeline_entries: int | None,
) -> dict:
    from google import genai
    from google.genai import types

    key = _gemini_api_key()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    data = video_path.read_bytes()
    mime = mimetypes.guess_type(str(video_path))[0]
    if not mime or not mime.startswith("video/"):
        mime = "video/quicktime" if video_path.suffix.lower() == ".mov" else "video/mp4"

    client = genai.Client(api_key=key)
    part = types.Part.from_bytes(data=data, mime_type=mime)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.15)
    response = client.models.generate_content(model=model, contents=[part, prompt], config=cfg)
    if not response.text:
        raise RuntimeError("empty Gemini response")
    return finalize_verdict(json.loads(response.text), expected_timeline_entries=expected_timeline_entries)


def call_gemini_video(
    *,
    model: str,
    video_path: Path,
    prompt: str,
    inline_video_max_mb: float = 19.0,
    expected_timeline_entries: int | None = None,
) -> dict:
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    key = _gemini_api_key()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY (Google AI Studio / Vertex-compatible key).")

    sz = video_path.stat().st_size
    cap = int(max(inline_video_max_mb, 0.5) * 1024 * 1024)
    if sz > cap:
        raise RuntimeError(
            f"Evaluated clip is {sz / (1024 * 1024):.2f} MiB; inline limit is {cap / (1024 * 1024):.1f} MiB "
            "(Gemini ~20 MiB request limit). Raise inline_video_max_mb, shorten max_duration_seconds, or use mode frames."
        )

    print(
        f"VLM: inline video ({sz / (1024 * 1024):.2f} MiB ≤ {cap / (1024 * 1024):.1f} MiB)…",
        flush=True,
    )
    try:
        return _call_gemini_video_inline_google_genai(
            video_path=video_path,
            model=model,
            prompt=prompt,
            expected_timeline_entries=expected_timeline_entries,
        )
    except ImportError as e:
        raise RuntimeError(
            "Inline video requires the `google-genai` package. "
            "pip install 'robot-manipulation-sim[vlm]'"
        ) from e


def call_gemini_frames(
    *,
    model: str,
    frame_paths: list[Path],
    prompt: str,
    expected_timeline_entries: int | None = None,
) -> dict:
    try:
        from PIL import Image
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: pip install 'robot-manipulation-sim[vlm]' (google-generativeai, pillow)"
        ) from e

    key = _gemini_api_key()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    genai.configure(api_key=key)
    parts: list[Any] = [prompt]
    for p in frame_paths:
        parts.append(Image.open(p))

    mdl = genai.GenerativeModel(
        model_name=model,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.15,
        ),
    )
    response = mdl.generate_content(parts)
    if not response.text:
        raise RuntimeError("empty Gemini response")
    return finalize_verdict(json.loads(response.text), expected_timeline_entries=expected_timeline_entries)


class VlmObserverAnalyzer:
    """Gemini judge: neutral observer stream + task-aligned evaluation when ``task_spec`` is configured."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def analyze(self, ctx: ValidationContext) -> AnalyzerResult:
        video = ctx.simulation.video
        if not video.is_file():
            return AnalyzerResult(
                "vlm_observer",
                ok=False,
                exit_code=2,
                messages=[f"video not found: {video}"],
            )

        mode = str(self.params.get("mode", "video"))
        max_dur = float(self.params.get("max_duration_seconds", 45.0))
        max_frames = int(self.params.get("max_frames", 10))
        model = str(self.params.get("model") or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
        inline_max_mb = float(self.params.get("inline_video_max_mb", os.environ.get("GEMINI_INLINE_VIDEO_MAX_MB", 19)))
        dry_run = coerce_bool(self.params.get("dry_run"), default=False)
        no_json = coerce_bool(self.params.get("no_json_file"), default=False)
        json_out_raw = self.params.get("json_out")
        json_out = Path(json_out_raw) if json_out_raw else None

        metrics_text = ctx.metrics_text
        task_spec = (ctx.task_spec or "").strip() or None

        if dry_run:
            msgs = self._dry_run(video, mode, max_dur, max_frames, model, no_json)
            return AnalyzerResult("vlm_observer", ok=True, exit_code=0, messages=msgs)

        if not _gemini_api_key():
            return AnalyzerResult(
                "vlm_observer",
                ok=False,
                exit_code=2,
                messages=[
                    "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Use dry_run: true in YAML to probe ffmpeg only."
                ],
            )

        try:
            if mode == "video":
                verdict, json_path = self._run_video(
                    video,
                    metrics_text,
                    task_spec,
                    max_dur,
                    model,
                    inline_max_mb,
                    no_json,
                    json_out,
                )
            elif mode == "frames":
                verdict, json_path = self._run_frames(
                    video,
                    metrics_text,
                    task_spec,
                    max_frames,
                    model,
                    no_json,
                    json_out,
                )
            else:
                return AnalyzerResult("vlm_observer", ok=False, exit_code=2, messages=[f"unknown mode: {mode!r}"])
        except Exception as exc:
            return AnalyzerResult(
                "vlm_observer",
                ok=False,
                exit_code=2,
                messages=[f"{type(exc).__name__}: {exc}"],
            )

        exit_code = 0 if bool(verdict.get("pass")) else 1
        msgs = []
        if json_path is not None:
            msgs.append(f"wrote {json_path.resolve()}")
        print(json.dumps(verdict, indent=2))
        return AnalyzerResult(
            "vlm_observer",
            ok=exit_code == 0,
            exit_code=exit_code,
            messages=msgs,
            artifacts={"verdict": verdict, "json_out": str(json_path) if json_path else None},
        )

    def _dry_run(
        self,
        video: Path,
        mode: str,
        max_dur: float,
        max_frames: int,
        model: str,
        no_json: bool,
    ) -> list[str]:
        msgs: list[str] = []
        full = probe_duration_seconds(video)
        if mode == "video":
            ev = min(full, max_dur)
            msgs.append(f"dry-run [video]: total={full:.2f}s, would evaluate first {ev:.2f}s, model={model}")
        else:
            mf = max(4, min(16, max_frames))
            msgs.append(f"dry-run [frames]: would sample up to {mf} PNGs, model={model}")
            with tempfile.TemporaryDirectory(prefix="vlm_dry_") as tmp:
                frames = extract_frames(video, Path(tmp), max_frames=mf)
                msgs.append(f"extracted {len(frames)} frame(s) for sanity")
                for f in frames[:3]:
                    msgs.append(f"  {f}")
        msgs.append("dry-run: skip Gemini")
        if not no_json:
            msgs.append(f"dry-run: on success, verdict would be written to {video.with_suffix('.vlm.json').resolve()}")
        for m in msgs:
            print(m, flush=True)
        return msgs

    def _run_video(
        self,
        video: Path,
        metrics_text: str | None,
        task_spec: str | None,
        max_dur: float,
        model: str,
        inline_max_mb: float,
        no_json: bool,
        json_out: Path | None,
    ) -> tuple[dict, Path | None]:
        with tempfile.TemporaryDirectory(prefix="vlm_rollout_") as tmp:
            media, full_dur, eval_dur = prepare_eval_video(video, Path(tmp), max_dur)
            print(
                f"evaluating video: {eval_dur:.1f}s of {full_dur:.1f}s total ({'clip' if media != video else 'original'})",
                flush=True,
            )
            expected_n = max(1, math.ceil(max(eval_dur, 0.05)))
            prompt = build_prompt(
                metrics_text=metrics_text,
                media_hint="You are given the rollout as an MP4 video (first content part below).",
                eval_duration_seconds=eval_dur,
                task_spec=task_spec,
            )
            verdict = call_gemini_video(
                model=model,
                video_path=media,
                prompt=prompt,
                inline_video_max_mb=inline_max_mb,
                expected_timeline_entries=expected_n,
            )
        json_path = self._write_json(verdict, video, no_json, json_out)
        return verdict, json_path

    def _run_frames(
        self,
        video: Path,
        metrics_text: str | None,
        task_spec: str | None,
        max_frames: int,
        model: str,
        no_json: bool,
        json_out: Path | None,
    ) -> tuple[dict, Path | None]:
        mf = max(4, min(16, max_frames))
        with tempfile.TemporaryDirectory(prefix="vlm_rollout_") as tmp:
            frames = extract_frames(video, Path(tmp), max_frames=mf)
            if not frames:
                raise RuntimeError("frame extraction produced no PNGs")
            print(f"frames mode: sampled {len(frames)} PNGs from {video}", flush=True)
            clip_dur = probe_duration_seconds(video)
            expected_n = max(1, math.ceil(max(clip_dur, 0.05)))
            prompt = build_prompt(
                metrics_text=metrics_text,
                media_hint=(
                    f"You are given {len(frames)} still images in chronological order, sampled approximately "
                    "uniformly in time across the clip; infer continuous motion when writing "
                    "**second_by_second_neutral** and **second_by_second_task**."
                ),
                eval_duration_seconds=clip_dur,
                task_spec=task_spec,
            )
            verdict = call_gemini_frames(
                model=model,
                frame_paths=frames,
                prompt=prompt,
                expected_timeline_entries=expected_n,
            )
        json_path = self._write_json(verdict, video, no_json, json_out)
        return verdict, json_path

    def _write_json(self, verdict: dict, video: Path, no_json: bool, json_out: Path | None) -> Path | None:
        if no_json:
            return None
        path = json_out if json_out is not None else video.with_suffix(".vlm.json")
        path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(f"wrote {path.resolve()}", flush=True)
        return path
