"""Generic ``joints.csv`` analyzer: trajectory progress and motion smoothness (velocity / accel / jerk).

Ignores actuator ``ctrl_*`` columns and metadata ``episode`` / ``sim_step`` / ``time_sec``. All other
scalar numeric columns are treated as configuration coordinates (joints, free-joint components, etc.).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.context import ValidationContext
from robot_manipulation_sim.validation.util import coerce_bool

_SKIP_KEYS = frozenset({"episode", "sim_step", "time_sec"})
_TIME = "time_sec"
_EPISODE = "episode"


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _value_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    out: list[str] = []
    for k in rows[0].keys():
        if k in _SKIP_KEYS or str(k).startswith("ctrl_"):
            continue
        try:
            float(rows[0].get(k, "") or "")
        except (TypeError, ValueError):
            continue
        out.append(str(k))
    return out


def _filter_episode(rows: list[dict[str, Any]], ep: int | None) -> list[dict[str, Any]]:
    if not rows or ep is None or _EPISODE not in rows[0]:
        return rows
    s = str(ep)
    return [r for r in rows if str(r.get(_EPISODE, "")).strip() == s]


def _n_seconds(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 1
    times = [float(r[_TIME]) for r in rows if r.get(_TIME) not in (None, "")]
    if not times:
        return 1
    return max(1, int(math.ceil(max(times) + 1e-9)))


def _bucket_rows(rows: list[dict[str, Any]], n_sec: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_sec)]
    for r in rows:
        try:
            si = int(math.floor(float(r[_TIME]) + 1e-9))
        except (TypeError, ValueError):
            continue
        si = max(0, min(n_sec - 1, si))
        buckets[si].append(r)
    for b in buckets:
        b.sort(key=lambda x: float(x.get(_TIME, 0.0) or 0.0))
    return buckets


def _derivatives(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return L2 norms per step for velocity, acceleration, jerk (each shorter than ``q``)."""
    if q.shape[0] < 4 or t.shape[0] != q.shape[0]:
        return np.array([]), np.array([]), np.array([])
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-9)
    vel = np.diff(q, axis=0) / dt[:, np.newaxis]
    if vel.shape[0] < 2:
        return np.linalg.norm(vel, axis=1), np.array([]), np.array([])
    dtm = 0.5 * (dt[:-1] + dt[1:])
    dtm = np.maximum(dtm, 1e-9)
    acc = np.diff(vel, axis=0) / dtm[:, np.newaxis]
    if acc.shape[0] < 2:
        vn = np.linalg.norm(vel, axis=1)
        an = np.linalg.norm(acc, axis=1)
        return vn, an, np.array([])
    dtm2 = 0.5 * (dtm[:-1] + dtm[1:])
    dtm2 = np.maximum(dtm2, 1e-9)
    jerk = np.diff(acc, axis=0) / dtm2[:, np.newaxis]
    return (
        np.linalg.norm(vel, axis=1),
        np.linalg.norm(acc, axis=1),
        np.linalg.norm(jerk, axis=1),
    )


def _error_verdict(error: str) -> dict[str, Any]:
    return {
        "analyzer": "joints_csv_trajectory",
        "pass": False,
        "summary_task_agnostic": error,
        "summary_task_evaluation": error,
        "second_by_second_neutral": [],
        "second_by_second_task": [],
        "issues": [error],
        "checks": {},
        "error": error,
    }


class JointsCsvTrajectoryAnalyzer:
    """Summarize joint-space motion: path length, vel / accel / jerk norms (generic, not task rubric)."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def _skip_json_file(self) -> bool:
        return coerce_bool(self.params.get("no_json_file"), default=False)

    def _resolve_json_path(self, ctx: ValidationContext) -> Path:
        jout = self.params.get("json_out")
        path_out = Path(jout) if jout else None
        vid = ctx.simulation.video
        stem = str(self.params.get("output_stem", "trajectory"))
        path_out = path_out or (vid.parent / f"{vid.stem}.{stem}.json")
        return path_out.resolve()

    def _persist_json(self, ctx: ValidationContext, verdict: dict[str, Any]) -> Path | None:
        if self._skip_json_file():
            return None
        path_out = self._resolve_json_path(ctx)
        path_out.parent.mkdir(parents=True, exist_ok=True)
        path_out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(f"joints_csv_trajectory: wrote {path_out}", flush=True)
        return path_out

    def _finish(
        self,
        ctx: ValidationContext,
        verdict: dict[str, Any],
        *,
        exit_code: int,
        ok: bool | None = None,
    ) -> AnalyzerResult:
        if ok is None:
            ok = exit_code == 0
        json_path = self._persist_json(ctx, verdict)
        print(json.dumps(verdict, indent=2), flush=True)
        return AnalyzerResult(
            "joints_csv_trajectory",
            ok=ok,
            exit_code=exit_code,
            messages=[str(json_path.resolve())] if json_path else [],
            artifacts={"verdict": verdict, "json_out": str(json_path) if json_path else None},
        )

    def analyze(self, ctx: ValidationContext) -> AnalyzerResult:
        jc = ctx.simulation.joints_csv
        if jc is None or not jc.is_file():
            return self._finish(ctx, _error_verdict("joints_csv missing"), exit_code=2, ok=False)

        ep = self.params.get("episode")
        ep_i = int(ep) if ep is not None else None
        rows = _filter_episode(_load(jc), ep_i)
        if len(rows) < 2:
            return self._finish(ctx, _error_verdict("not enough rows"), exit_code=2, ok=False)

        cols = _value_columns(rows)
        if not cols:
            return self._finish(ctx, _error_verdict("no numeric joint columns found"), exit_code=2, ok=False)

        t = np.array([float(r[_TIME]) for r in rows], dtype=np.float64)
        q = np.array([[float(r.get(c, 0) or 0.0) for c in cols] for r in rows], dtype=np.float64)

        dq = np.diff(q, axis=0)
        path_len = float(np.sum(np.linalg.norm(dq, axis=1)))

        vn, an, jn = _derivatives(t, q)
        max_vel = float(np.max(vn)) if vn.size else 0.0
        max_acc = float(np.max(an)) if an.size else 0.0
        max_jerk = float(np.max(jn)) if jn.size else 0.0
        rms_vel = float(np.sqrt(np.mean(vn**2))) if vn.size else 0.0
        rms_acc = float(np.sqrt(np.mean(an**2))) if an.size else 0.0
        rms_jerk = float(np.sqrt(np.mean(jn**2))) if jn.size else 0.0

        n_sec = int(self.params.get("max_seconds", 0) or 0) or _n_seconds(rows)
        n_sec = max(1, min(n_sec, 7200))
        buckets = _bucket_rows(rows, n_sec)

        sec_lines_n: list[str] = []
        sec_lines_t: list[str] = []
        for si, b in enumerate(buckets):
            if not b:
                sec_lines_n.append(f"[{si},{si+1})s: no samples")
                sec_lines_t.append(f"[{si},{si+1})s: no samples")
                continue
            tb = np.array([float(r[_TIME]) for r in b], dtype=np.float64)
            qb = np.array([[float(r.get(c, 0) or 0.0) for c in cols] for r in b], dtype=np.float64)
            vnb, _, jnb = _derivatives(tb, qb)
            mv = float(np.mean(vnb)) if vnb.size else 0.0
            mj = float(np.mean(jnb)) if jnb.size else 0.0
            sec_lines_n.append(f"[{si},{si+1})s: mean|v|_L2={mv:.4f} mean|jerk|_L2={mj:.4f} n={len(b)}")
            ts = (ctx.task_spec or "").strip()
            sec_lines_t.append(
                f"[{si},{si+1})s: smoothness sample (mean|v|={mv:.4f}, mean|jerk|={mj:.4f}); "
                f"task excerpt: {ts[:80]!r}{'...' if len(ts) > 80 else ''}"
                if ts
                else f"[{si},{si+1})s: smoothness sample (mean|v|={mv:.4f}, mean|jerk|={mj:.4f})."
            )

        max_rms_jerk = self.params.get("max_rms_jerk")
        max_peak_jerk = self.params.get("max_peak_jerk")
        ok_jerk = True
        issues: list[str] = []
        if max_rms_jerk is not None and rms_jerk > float(max_rms_jerk):
            ok_jerk = False
            issues.append(f"rms jerk {rms_jerk:.4f} exceeds limit {float(max_rms_jerk):.4f}")
        if max_peak_jerk is not None and max_jerk > float(max_peak_jerk):
            ok_jerk = False
            issues.append(f"peak jerk {max_jerk:.4f} exceeds limit {float(max_peak_jerk):.4f}")

        sum_neutral = (
            f"rows={len(rows)} cols={len(cols)} path_len_L2={path_len:.4f}; "
            f"peak |v|_L2={max_vel:.4f} peak |a|_L2={max_acc:.4f} peak |j|_L2={max_jerk:.4f}; "
            f"rms |v|={rms_vel:.4f} rms |a|={rms_acc:.4f} rms |j|={rms_jerk:.4f}."
        )
        ts = (ctx.task_spec or "").strip()
        sum_eval = (
            "Generic trajectory metrics (not a task pass/fail unless limits set in params). "
            f"{sum_neutral} Task excerpt: {ts[:160]!r}{'...' if len(ts) > 160 else ''}"
            if ts
            else "Generic trajectory metrics; see neutral summary and checks."
        )

        verdict: dict[str, Any] = {
            "analyzer": "joints_csv_trajectory",
            "pass": bool(ok_jerk),
            "summary_task_agnostic": sum_neutral,
            "summary_task_evaluation": sum_eval,
            "second_by_second_neutral": sec_lines_n,
            "second_by_second_task": sec_lines_t,
            "issues": issues,
            "checks": {
                "path_length_l2": path_len,
                "peak_vel_l2": max_vel,
                "peak_acc_l2": max_acc,
                "peak_jerk_l2": max_jerk,
                "rms_vel_l2": rms_vel,
                "rms_acc_l2": rms_acc,
                "rms_jerk_l2": rms_jerk,
                "n_config_coords": len(cols),
            },
        }

        exit_code = 0 if verdict["pass"] else 1
        return self._finish(ctx, verdict, exit_code=exit_code)
