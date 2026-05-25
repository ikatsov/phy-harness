"""Task-specific joint CSV analyzer: slow shoulder_pan only, other joints near episode start (UR5 harness).

Not a generic joint-log tool — hard-coded columns and rubric for the **base rotation at home** task.
When the user task changes, add a **new** small module under ``task_specific/`` and a
``TASK_SPECIFIC_REGISTRY`` entry (see robot-manipulation-task-loop skill). Do **not** put task rubrics
in ``generic/``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.context import ValidationContext
from robot_manipulation_sim.validation.util import coerce_bool

PAN = "shoulder_pan_joint"
NON_PAN = (
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Legacy parallel gripper column, or Robotiq driver joint (opening proxy) in Menagerie 2F-85 logs.
GRIPPER_SLIDE = "gripper_slide"
GRIPPER_ROBOTIQ_DRIVER = "left_driver_joint"
BOX_Z = "grasp_box_free.pz"
TIME = "time_sec"
EPISODE = "episode"


def _gripper_col(row0: dict[str, Any]) -> tuple[str | None, str]:
    """Column name for gripper motion in joints.csv, and unit label for messages."""
    if GRIPPER_SLIDE in row0:
        return GRIPPER_SLIDE, "m"
    if GRIPPER_ROBOTIQ_DRIVER in row0:
        return GRIPPER_ROBOTIQ_DRIVER, "rad"
    return None, ""


def _f(row: dict[str, Any], k: str, d: float = 0.0) -> float:
    v = row.get(k)
    if v is None or v == "":
        return d
    return float(v)


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _filter_episode(rows: list[dict[str, Any]], ep: int | None) -> list[dict[str, Any]]:
    if not rows or ep is None or EPISODE not in rows[0]:
        return rows
    s = str(ep)
    return [r for r in rows if str(r.get(EPISODE, "")).strip() == s]


def _n_seconds(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 1
    return max(1, int(math.ceil(max(_f(r, TIME) for r in rows) + 1e-9)))


def _buckets(rows: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for r in rows:
        sec = int(math.floor(_f(r, TIME) + 1e-9))
        sec = max(0, min(n - 1, sec))
        out[sec].append(r)
    for b in out:
        b.sort(key=lambda x: _f(x, TIME))
    return out


def _span(bucket: list[dict[str, Any]], col: str) -> float:
    if not bucket or col not in bucket[0]:
        return 0.0
    xs = [_f(r, col) for r in bucket]
    return max(xs) - min(xs)


def _neutral_line(si: int, b: list[dict[str, Any]]) -> str:
    if not b:
        return f"[{si},{si+1})s: no samples."
    lo, hi, sp = (min(_f(x, PAN) for x in b), max(_f(x, PAN) for x in b), _span(b, PAN))
    bits = [f"pan[{lo:.4f},{hi:.4f}] d={sp:.4f}rad"]
    for c in NON_PAN:
        s = _span(b, c)
        if s > 1e-5:
            bits.append(f"{c.replace('_joint','')}_d={s:.4f}")
    gcol, gunit = _gripper_col(b[0])
    if gcol:
        bits.append(f"grip_d={_span(b, gcol):.5f}{gunit}")
    if BOX_Z in b[0]:
        bits.append(f"box_z_d={_span(b, BOX_Z):.4f}m")
    return "; ".join(bits[:7])


def _task_line(b: list[dict[str, Any]], span_lim: float) -> str:
    if not b:
        return "no data."
    arm = max(_span(b, c) for c in NON_PAN if c in b[0])
    gcol, _ = _gripper_col(b[0])
    g = _span(b, gcol) if gcol else 0.0
    if gcol == GRIPPER_SLIDE:
        g_fail = g > 5e-4
    elif gcol:
        g_fail = g > 0.03
    else:
        g_fail = False
    pp = _span(b, PAN)
    if arm > span_lim or g_fail:
        return f"FAIL slice: non-pan span={arm:.4f}rad (lim {span_lim:.4f}) grip_d={g:.5f}"
    if pp < 1e-4:
        return f"WARN slice: little pan motion (d={pp:.4f}rad)"
    return f"OK slice: pan_d={pp:.4f}rad arm_span={arm:.4f}rad"


class JointsCsvBaseRotationAnalyzer:
    """Rubric: net |pan| >= min_pan_total_rad; non-pan joints stay within home_max_excursion_rad of t0; per-s non-pan span <= home_max_span_per_s."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def analyze(self, ctx: ValidationContext) -> AnalyzerResult:
        jc = ctx.simulation.joints_csv
        if jc is None or not jc.is_file():
            return AnalyzerResult(
                "joints_csv_base_rotation",
                ok=False,
                exit_code=2,
                messages=["joints_csv missing"],
            )

        ep = self.params.get("episode")
        ep_i = int(ep) if ep is not None else None
        span_lim = float(self.params.get("home_max_span_per_s", 0.06))
        home_exc = float(self.params.get("home_max_excursion_rad", 0.12))
        min_pan = float(self.params.get("min_pan_total_rad", math.pi))

        rows = _filter_episode(_load(jc), ep_i)
        if not rows:
            return AnalyzerResult("joints_csv_base_rotation", ok=False, exit_code=2, messages=["no rows"])

        n = int(self.params.get("max_seconds", 0) or 0) or _n_seconds(rows)
        n = max(1, min(n, 7200))
        buckets = _buckets(rows, n)
        t0 = rows[0]

        pan_net = abs(_f(rows[-1], PAN) - _f(t0, PAN))
        max_exc = 0.0
        for c in NON_PAN:
            if c not in t0:
                continue
            q0 = _f(t0, c)
            for r in rows:
                max_exc = max(max_exc, abs(_f(r, c) - q0))

        sec_n = [_neutral_line(i, buckets[i]) for i in range(n)]
        sec_t = [_task_line(buckets[i], span_lim) for i in range(n)]

        ok_pan = pan_net >= min_pan
        ok_home = max_exc <= home_exc
        bad_slice = any("FAIL" in sec_t[i] for i in range(n))

        ts = (ctx.task_spec or "").strip()
        sum_eval = (
            f"Base-rotation rubric (joints only): net |pan|={pan_net:.3f} rad (need >= {min_pan:.3f}); "
            f"max non-pan |q-q0(t0)|={max_exc:.3f} rad (limit {home_exc:.3f}). "
            f"Task text excerpt: {ts[:200]!r}{'...' if len(ts) > 200 else ''}"
            if ts
            else "No task_spec in YAML; numeric rubric applied standalone."
        )

        sum_neutral = (
            f"Rows={len(rows)} t=[{_f(rows[0], TIME):.2f},{_f(rows[-1], TIME):.2f}]s; "
            f"net pan={pan_net:.4f}rad; max non-pan excursion from t0={max_exc:.4f}rad; "
            f"seconds={n}. Columns: pan+{','.join(NON_PAN)}."
        )

        verdict: dict[str, Any] = {
            "analyzer": "joints_csv_base_rotation",
            "pass": bool(ok_pan and ok_home and not bad_slice),
            "summary_task_agnostic": sum_neutral,
            "summary_task_evaluation": sum_eval,
            "second_by_second_neutral": sec_n,
            "second_by_second_task": sec_t,
            "issues": [],
            "checks": {
                "pan_net_rad": pan_net,
                "min_pan_total_rad": min_pan,
                "max_non_pan_excursion_rad": max_exc,
                "home_max_excursion_rad": home_exc,
                "home_max_span_per_s": span_lim,
                "any_slice_fail": bad_slice,
            },
        }
        if not ok_pan:
            verdict["issues"].append(f"net pan {pan_net:.4f} < required {min_pan:.4f} rad")
        if not ok_home:
            verdict["issues"].append(f"non-pan excursion {max_exc:.4f} > {home_exc:.4f} rad")
        if bad_slice:
            verdict["issues"].append("one or more 1s windows exceeded per-second non-pan span")

        no_json = coerce_bool(self.params.get("no_json_file"), default=False)
        jout = self.params.get("json_out")
        path_out = Path(jout) if jout else None
        json_path: Path | None = None
        if not no_json:
            vid = ctx.simulation.video
            stem = self.params.get("output_stem", "joints_base")
            path_out = path_out or (vid.parent / f"{vid.stem}.{stem}.json")
            path_out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
            json_path = path_out
            print(f"joints_csv_base_rotation: wrote {path_out.resolve()}", flush=True)

        print(json.dumps(verdict, indent=2))
        exit_code = 0 if verdict["pass"] else 1
        return AnalyzerResult(
            "joints_csv_base_rotation",
            ok=exit_code == 0,
            exit_code=exit_code,
            messages=[str(json_path.resolve())] if json_path else [],
            artifacts={"verdict": verdict, "json_out": str(json_path) if json_path else None},
        )
