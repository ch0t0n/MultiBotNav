#!/usr/bin/env python3
"""
check_status.py

One-stop "status & preflight" tool for this repo.

This consolidates (and extends) the following helper scripts:
  - check_tuning_trials.py / check_tuning_trials_strict.py
  - preflight_train_from_tuned.py / before_train_from_tuned.py
  - before_transfer_check.py
  - inspect_globs.py
  - (optional) sync_failed_runs_from_slurm.py capabilities via `sync_failures`

Design goals
------------
- NO CSV output (prints human-readable, actionable summaries).
- Safe by default: read-only checks (except `sync_failures --write`, which appends).
- Optimized for diagnosing failures and understanding run progress:
    * shows missing prerequisites
    * shows per-run completion (trained_model.zip)
    * shows last logged progress from tensorboard/progress.csv when available
    * shows latest failure from failed_runs.jsonl when available
    * prints concrete sbatch --array suggestions for reruns

Typical usage (from repo root)
------------------------------
# 1) Is tuning complete?
python check_status.py tuning --version v2

# 2) Before launching train-from-tuned arrays:
python check_status.py before_train_from_tuned --only v2

# 3) Before launching transfer arrays:
python check_status.py before_transfer --only v2 --load_set 1

# 4) Check what plotting/table scripts will glob:
python check_status.py globs --v2-seed 0 --algo-counts

# 5) Summarize failures captured by append_failure / slurm scan:
python check_status.py failures --tail 25

# 6) (Optional) Scan Slurm out/err logs and append NEW failures:
python check_status.py sync_failures --print_new   # dry-run (default)
python check_status.py sync_failures --write        # append new entries
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import socket
import sys
import time
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Optional deps
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

try:
    import fcntl  # Unix-only
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


# -------------------------
# Small formatting helpers
# -------------------------

def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fmt_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "?"


def _fmt_age_seconds(age_s: float) -> str:
    if age_s < 0:
        age_s = 0.0
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{age_s/60:.1f}m"
    if age_s < 86400:
        return f"{age_s/3600:.1f}h"
    return f"{age_s/86400:.1f}d"


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _truncate(s: str, n: int = 140) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _human_int(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        if float(x).is_integer():
            return f"{int(float(x)):,}"
        return f"{float(x):,.3f}"
    except Exception:
        return str(x)


# -------------------------
# Bash parsing helpers
# -------------------------

@dataclass(frozen=True)
class ScriptArrays:
    path: Path
    algorithms: List[str]
    sets: List[int]
    seeds: Optional[List[int]]  # None for v0-style scripts (seed passed as scalar)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_bash_array(text: str, var: str) -> Optional[List[str]]:
    """
    Supports:
      algorithms=("A2C" "PPO")
      algorithms=(A2C PPO)
    with optional whitespace and multi-line arrays.
    """
    m = re.search(rf"{re.escape(var)}\s*=\s*\((.*?)\)", text, flags=re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    toks = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\S+)', body)
    out: List[str] = []
    for a, b, c in toks:
        val = a or b or c
        if not val:
            continue
        if val.startswith("#"):
            continue
        out.append(val)
    return out or None


def _parse_cli_token(text: str, flag: str) -> Optional[str]:
    """
    Parse a CLI token value from a bash script.

    Supports:
      --flag value
      --flag=value
      --flag "value"
      --flag 'value'
    """
    m = re.search(rf"{re.escape(flag)}=(?P<val>(\"[^\"]*\"|'[^']*'|\S+))", text)
    if not m:
        m = re.search(rf"{re.escape(flag)}\s+(?P<val>(\"[^\"]*\"|'[^']*'|\S+))", text)
    if not m:
        return None
    val = m.group("val").strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1]
    return val


def _parse_cli_int(text: str, flag: str) -> Optional[int]:
    tok = _parse_cli_token(text, flag)
    if tok is None:
        return None
    if "$" in tok:
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def _load_arrays(script_path: Path) -> ScriptArrays:
    text = _read_text(script_path)
    algos = _parse_bash_array(text, "algorithms")
    sets = _parse_bash_array(text, "sets")
    seeds = _parse_bash_array(text, "seeds")

    if not algos or not sets:
        raise ValueError(f"Could not parse algorithms/sets from: {script_path}")

    out_seeds: Optional[List[int]] = None
    if seeds:
        out_seeds = [int(x) for x in seeds]

    return ScriptArrays(
        path=script_path,
        algorithms=[a.strip() for a in algos],
        sets=[int(s) for s in sets],
        seeds=out_seeds,
    )


def _compute_array_index(arr: ScriptArrays, algorithm: str, set_id: int, seed: Optional[int]) -> int:
    alg_idx = arr.algorithms.index(algorithm)
    set_idx = arr.sets.index(set_id)
    if arr.seeds is None:
        return alg_idx * len(arr.sets) + set_idx
    if seed is None:
        raise ValueError("seed is required for scripts with a seeds array")
    seed_idx = arr.seeds.index(seed)
    return alg_idx * (len(arr.sets) * len(arr.seeds)) + set_idx * len(arr.seeds) + seed_idx


# -------------------------
# File validity helpers
# -------------------------

def _is_good_zip(p: Path) -> Tuple[bool, int, bool]:
    if not p.is_file():
        return (False, 0, False)
    try:
        size = p.stat().st_size
    except Exception:
        size = 0
    try:
        iszip = zipfile.is_zipfile(p)
    except Exception:
        iszip = False
    ok = (size > 0) and iszip
    return (ok, size, iszip)


def _yaml_valid_bestparams(obj: Any) -> Tuple[bool, str]:
    """
    Validate the structure expected by train_v2.py:
      - tune_v2.py writes: {"filtered_params": {...}, ...}
      - tune.py (old) writes: {...} (direct dict)
    train_v2.py supports both via payload.get("filtered_params", payload).
    """
    if not isinstance(obj, dict) or not obj:
        return False, "not_a_dict_or_empty"

    if "filtered_params" in obj:
        fp = obj.get("filtered_params")
        if isinstance(fp, dict) and fp:
            return True, "v2_filtered_params"
        return False, "v2_filtered_params_missing_or_empty"

    return True, "v0_dict"


def _read_yaml_if_possible(p: Path) -> Tuple[Optional[Any], Optional[str]]:
    if yaml is None:
        return None, "PyYAML not installed"
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8", errors="replace")), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _find_latest_backup_bestparams(root: Path, study_name: str) -> Optional[Path]:
    base = root / "logs" / "tuning_logs"
    hits = sorted(base.glob(f"{study_name}__old_*/best_hyperparameters.yaml"))
    return hits[-1] if hits else None


# -------------------------
# Progress parsing helpers
# -------------------------

def _tail_last_nonempty_line(path: Path, *, max_bytes: int = 256_000) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        size = path.stat().st_size
        if size == 0:
            return None
        with path.open("rb") as f:
            read_size = min(size, max_bytes)
            f.seek(-read_size, os.SEEK_END)
            data = f.read(read_size)
        text = data.decode("utf-8", errors="replace")
        lines = [ln.strip("\r") for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        # If file is huge, the first line in our tail chunk might be partial; that's fine for "last line".
        return lines[-1]
    except Exception:
        return None


def _read_first_line(path: Path, *, max_bytes: int = 64_000) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        text = data.decode("utf-8", errors="replace")
        # First non-empty line
        for ln in text.splitlines():
            if ln.strip():
                return ln.strip("\r")
        return None
    except Exception:
        return None


def _parse_progress_csv(progress_csv: Path) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Parse the last row of stable-baselines3 logger's progress.csv.

    Returns: (dict, error_str)
    """
    header_line = _read_first_line(progress_csv)
    if not header_line:
        return None, "empty_or_missing_header"
    last_line = _tail_last_nonempty_line(progress_csv)
    if not last_line or last_line == header_line:
        return None, "no_rows"
    try:
        header = next(csv.reader([header_line]))
        row = next(csv.reader([last_line]))
        if len(row) != len(header):
            # Best-effort: ignore mismatch
            return None, f"row_len_mismatch header={len(header)} row={len(row)}"
        return dict(zip(header, row)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _extract_progress_metrics(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    """
    Pull the most useful "at a glance" metrics if present.
    """
    def _get_float(*keys: str) -> Optional[float]:
        for k in keys:
            if k in row and row[k] not in ("", "nan", "NaN", "None"):
                try:
                    return float(row[k])
                except Exception:
                    continue
        return None

    return {
        "total_timesteps": _get_float("time/total_timesteps", "total_timesteps"),
        "ep_rew_mean": _get_float("rollout/ep_rew_mean", "train/ep_rew_mean", "ep_rew_mean"),
        "fps": _get_float("time/fps", "fps"),
        "time_elapsed": _get_float("time/time_elapsed", "time_elapsed"),
        "eval_mean_reward": _get_float("eval/mean_reward", "mean_reward"),
    }


@dataclass
class RunStatus:
    run_name: str
    run_dir: Path
    is_present: bool
    trained_model_ok: bool
    trained_model_path: Path
    trained_model_size: int
    trained_model_iszip: bool
    progress_csv: Optional[Path]
    last_progress: Optional[Dict[str, Optional[float]]]
    last_update_ts: Optional[float]
    last_update_source: str
    note: str = ""


def _summarize_run_dir(run_name: str, run_dir: Path, *, trained_model_rel: str = "checkpoints/trained_model.zip") -> RunStatus:
    trained_model_path = run_dir / trained_model_rel
    trained_ok, size, iszip = _is_good_zip(trained_model_path)

    progress_csv = run_dir / "tensorboard" / "progress.csv"
    last_progress: Optional[Dict[str, Optional[float]]] = None
    note = ""
    last_update_ts: Optional[float] = None
    last_update_source = ""

    # Prefer progress.csv mtime; else fall back to newest file in run_dir (shallow).
    if progress_csv.exists() and progress_csv.is_file():
        row, err = _parse_progress_csv(progress_csv)
        if row is not None:
            last_progress = _extract_progress_metrics(row)
        else:
            note = f"progress.csv:{err}"
        try:
            last_update_ts = progress_csv.stat().st_mtime
            last_update_source = "progress.csv"
        except Exception:
            pass
    else:
        # Try tensorboard event dirs (shallow)
        tb_dir = run_dir / "tensorboard"
        newest = 0.0
        newest_src = ""
        if tb_dir.exists() and tb_dir.is_dir():
            try:
                for p in tb_dir.rglob("events.out.tfevents.*"):
                    if p.is_file():
                        mt = p.stat().st_mtime
                        if mt > newest:
                            newest = mt
                            newest_src = _rel(run_dir, p)
            except Exception:
                pass
        if newest > 0:
            last_update_ts = newest
            last_update_source = newest_src or "event_file"
        else:
            try:
                last_update_ts = run_dir.stat().st_mtime
                last_update_source = "dir_mtime"
            except Exception:
                pass

    return RunStatus(
        run_name=run_name,
        run_dir=run_dir,
        is_present=run_dir.exists(),
        trained_model_ok=trained_ok,
        trained_model_path=trained_model_path,
        trained_model_size=size,
        trained_model_iszip=iszip,
        progress_csv=progress_csv if progress_csv.exists() else None,
        last_progress=last_progress,
        last_update_ts=last_update_ts,
        last_update_source=last_update_source,
        note=note,
    )


# -------------------------
# failed_runs.jsonl helpers
# -------------------------

@dataclass
class FailureInfo:
    ts: str
    scheme: str
    script: str
    run_name: Optional[str]
    exc_type: Optional[str]
    exc: Optional[str]
    reason: Optional[str]
    slurm: Dict[str, Any]

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "FailureInfo":
        return FailureInfo(
            ts=str(obj.get("ts") or ""),
            scheme=str(obj.get("scheme") or ""),
            script=str(obj.get("script") or ""),
            run_name=obj.get("run_name"),
            exc_type=obj.get("exc_type"),
            exc=obj.get("exc"),
            reason=obj.get("reason"),
            slurm=obj.get("slurm") or {},
        )


def _load_failures_latest(failed_runs_path: Path, *, max_lines: int = 200_000) -> Tuple[Dict[str, FailureInfo], Counter]:
    """
    Returns:
      - latest_by_run_name: run_name -> latest FailureInfo
      - counts_by_scheme: Counter
    """
    latest: Dict[str, FailureInfo] = {}
    counts: Counter = Counter()

    if not failed_runs_path.exists():
        return latest, counts

    # Read in streaming fashion; keep latest by comparing ISO timestamps lexicographically (works for ISO8601).
    n = 0
    with failed_runs_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if n > max_lines:
                # Avoid pathological huge files on login nodes.
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fi = FailureInfo.from_json(obj)
            counts[fi.scheme] += 1
            if fi.run_name:
                prev = latest.get(fi.run_name)
                if prev is None or fi.ts >= prev.ts:
                    latest[fi.run_name] = fi

    return latest, counts


def _fmt_failure_brief(fi: FailureInfo) -> str:
    bits = [fi.ts, fi.scheme]
    if fi.exc_type:
        bits.append(fi.exc_type)
    if fi.reason:
        bits.append(fi.reason)
    if fi.exc:
        bits.append(_truncate(str(fi.exc), 120))
    return " | ".join(bits)


# -------------------------
# Naming helpers
# -------------------------

def _study_name_v0(algorithm: str, set_id: int, seed: int) -> str:
    return f"{algorithm}_set{set_id}_seed{seed}_v0"


def _name_v2(algorithm: str, set_id: int, seed: int, version_tag: Optional[int]) -> str:
    if version_tag is None:
        return f"{algorithm}_set{set_id}_seed{seed}"
    return f"{algorithm}_set{set_id}_seed{seed}_{version_tag}"


def _transfer_name_v0(algorithm: str, load_set: int, train_set: int, seed: int) -> str:
    return f"{algorithm}_from{load_set}_to{train_set}_seed{seed}_v0"


def _transfer_name_v2(algorithm: str, load_set: int, train_set: int, seed: int, version_tag: Optional[int]) -> str:
    if version_tag is None:
        return f"{algorithm}_from{load_set}_to{train_set}_seed{seed}"
    return f"{algorithm}_from{load_set}_to{train_set}_seed{seed}_{version_tag}"


# -------------------------
# Command: tuning
# -------------------------

TRIAL_RE = re.compile(r"^trial_(\d+)$")


@dataclass(frozen=True)
class StudyStatus:
    study_name: str
    expected_trials: int
    found_trials: int
    missing_trials: List[int]
    extra_trials: List[int]
    trials_missing_events: List[int]
    bestparams_ok: bool
    bestparams_reason: str
    bestparams_path: Path
    bestmodel_ok: bool
    bestmodel_path: Path
    has_study_config: bool
    last_update_ts: Optional[float]


def _study_matches_version(study_name: str, version: str) -> bool:
    if version == "all":
        return True
    if version == "v0":
        return study_name.endswith("_v0")
    if version == "v2":
        # Accept "_<digits>" suffix (not v0).
        return (not study_name.endswith("_v0")) and re.search(r"_\d+$", study_name) is not None
    return False


def _load_expected_trials(study_dir: Path, default_expected: int) -> Tuple[int, bool]:
    cfg = study_dir / "study_config.yaml"
    if yaml is None or not cfg.exists():
        return default_expected, False
    obj, err = _read_yaml_if_possible(cfg)
    if err is None and isinstance(obj, dict):
        trials = obj.get("trials")
        if isinstance(trials, int) and trials > 0:
            return trials, True
    return default_expected, False


def _trial_numbers(trials_dir: Path) -> Set[int]:
    nums: Set[int] = set()
    if not trials_dir.is_dir():
        return nums
    for child in trials_dir.iterdir():
        if not child.is_dir():
            continue
        m = TRIAL_RE.match(child.name)
        if m:
            nums.add(int(m.group(1)))
    return nums


def _trial_has_event_file(trial_dir: Path) -> bool:
    tb = trial_dir / "tensorboard"
    if not tb.exists():
        return False
    try:
        for p in tb.rglob("events.out.tfevents.*"):
            if p.is_file() and p.stat().st_size > 0:
                return True
    except Exception:
        return False
    return False


def _compute_last_update_ts(study_dir: Path, found_trials: Set[int]) -> Optional[float]:
    """
    Best-effort 'last touched' time for a tuning study, without heavy recursive walks.
    """
    candidates: List[Path] = [
        study_dir / "best_hyperparameters.yaml",
        study_dir / "best_model.zip",
        study_dir / "study_config.yaml",
    ]
    ts = 0.0
    for p in candidates:
        if p.exists():
            try:
                ts = max(ts, p.stat().st_mtime)
            except Exception:
                pass

    trials_dir = study_dir / "trials"
    for t in found_trials:
        td = trials_dir / f"trial_{t:03d}"
        if td.exists():
            try:
                ts = max(ts, td.stat().st_mtime)
            except Exception:
                pass
    return ts if ts > 0 else None


def _scan_study(study_dir: Path, *, version: str, default_expected: int, require_events: bool) -> Optional[StudyStatus]:
    study_name = study_dir.name
    if "__old_" in study_name:
        return None
    if not _study_matches_version(study_name, version):
        return None

    expected, has_cfg = _load_expected_trials(study_dir, default_expected)
    trials_dir = study_dir / "trials"
    found = _trial_numbers(trials_dir)

    missing = sorted(set(range(expected)) - found)
    extra = sorted(found - set(range(expected)))

    # best_hyperparameters.yaml
    bestparams_path = study_dir / "best_hyperparameters.yaml"
    bestparams_ok = False
    bestparams_reason = ""
    if bestparams_path.exists() and bestparams_path.is_file():
        size = bestparams_path.stat().st_size
        if size <= 0:
            bestparams_ok = False
            bestparams_reason = "empty"
        elif yaml is None:
            # Cannot validate structure, but file is non-empty.
            bestparams_ok = True
            bestparams_reason = "present_no_yaml_validation"
        else:
            obj, err = _read_yaml_if_possible(bestparams_path)
            if err is not None:
                bestparams_ok = False
                bestparams_reason = f"yaml_error:{err}"
            else:
                ok, mode = _yaml_valid_bestparams(obj)
                bestparams_ok = ok
                bestparams_reason = mode
    else:
        bestparams_ok = False
        bestparams_reason = "missing"

    # best_model.zip
    bestmodel_path = study_dir / "best_model.zip"
    bestmodel_ok, _size, _iszip = _is_good_zip(bestmodel_path)

    # trial event files
    trials_missing_events: List[int] = []
    if require_events:
        for t in sorted(found):
            td = trials_dir / f"trial_{t:03d}"
            if not _trial_has_event_file(td):
                trials_missing_events.append(t)

    last_update_ts = _compute_last_update_ts(study_dir, found)

    return StudyStatus(
        study_name=study_name,
        expected_trials=expected,
        found_trials=len(found),
        missing_trials=missing,
        extra_trials=extra,
        trials_missing_events=trials_missing_events,
        bestparams_ok=bestparams_ok,
        bestparams_reason=bestparams_reason,
        bestparams_path=bestparams_path,
        bestmodel_ok=bestmodel_ok,
        bestmodel_path=bestmodel_path,
        has_study_config=has_cfg,
        last_update_ts=last_update_ts,
    )


def cmd_tuning(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    tuning_root = root / args.tuning_dir
    if not tuning_root.exists():
        print(f"ERROR: tuning directory not found: {tuning_root}", file=sys.stderr)
        return 2

    latest_fail, scheme_counts = _load_failures_latest(root / args.failed_runs) if args.failures else ({}, Counter())

    statuses: List[StudyStatus] = []
    for sd in sorted([p for p in tuning_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        st = _scan_study(sd, version=args.version, default_expected=args.expected, require_events=args.require_events)
        if st is not None:
            statuses.append(st)

    if not statuses:
        print("No studies found that match your filters.")
        return 0

    def _is_complete(s: StudyStatus) -> bool:
        if s.missing_trials:
            return False
        if args.require_events and s.trials_missing_events:
            return False
        if not s.bestparams_ok:
            return False
        if args.require_bestmodel and not s.bestmodel_ok:
            return False
        return True

    complete = [s for s in statuses if _is_complete(s)]
    incomplete = [s for s in statuses if not _is_complete(s)]

    _print_header(f"TUNING STATUS  (root={root})  ({_now_local_iso()})")
    print(f"Scanned studies: {len(statuses)}  Complete: {len(complete)}  Incomplete: {len(incomplete)}")
    if yaml is None:
        print("[WARN] PyYAML is not installed; best_hyperparameters.yaml structure cannot be validated.")

    # Reason breakdown (for incomplete)
    reasons = Counter()
    for s in incomplete:
        if s.missing_trials:
            reasons["missing_trials"] += 1
        if s.extra_trials:
            reasons["extra_trials"] += 1
        if args.require_events and s.trials_missing_events:
            reasons["trials_missing_events"] += 1
        if not s.bestparams_ok:
            reasons[f"bestparams:{s.bestparams_reason}"] += 1
        if args.require_bestmodel and not s.bestmodel_ok:
            reasons["missing_or_bad_best_model_zip"] += 1
    if reasons:
        print("\nTop incomplete reasons:")
        for k, v in reasons.most_common(12):
            print(f"  {k}: {v}")

    if incomplete:
        print(f"\nIncomplete studies (showing up to {args.show}):")
        now = time.time()
        for s in incomplete[: args.show]:
            age = _fmt_age_seconds(now - s.last_update_ts) if s.last_update_ts else "-"
            parts = [f"- {s.study_name}: {s.found_trials}/{s.expected_trials} trials (age {age})"]
            if s.missing_trials:
                parts.append(f"missing={_truncate(str(s.missing_trials[:10]), 60)}{'…' if len(s.missing_trials)>10 else ''}")
            if s.extra_trials:
                parts.append(f"extra={len(s.extra_trials)}")
            if args.require_events and s.trials_missing_events:
                parts.append(f"no_events={len(s.trials_missing_events)}")
            if not s.bestparams_ok:
                parts.append(f"bestparams={s.bestparams_reason}")
            if args.require_bestmodel and not s.bestmodel_ok:
                parts.append("best_model.zip=BAD/MISSING")
            line = " | ".join(parts)
            print(line)
            if args.failures:
                fi = latest_fail.get(s.study_name)
                if fi:
                    print(f"    latest_failure: {_fmt_failure_brief(fi)}")

    return 2 if incomplete else 0


# -------------------------
# Command: before_train_from_tuned
# -------------------------

@dataclass(frozen=True)
class ExpectedBestparams:
    version: str            # "v0" | "v2"
    script: Path
    algorithm: str
    set_id: int
    train_seed: int
    version_tag: Optional[int]
    study_name: str
    bestparams_path: Path
    bestmodel_path: Path
    tuned_params_literal: Optional[str]
    reuse_seed_suggestion: Optional[Path]


def _iter_expected_from_train_from_tuned(
    root: Path,
    *,
    only: str,
    v0_seed_default: int,
    suggest_reuse_seed: Optional[int],
) -> Iterable[ExpectedBestparams]:
    scripts: List[Tuple[str, Path]] = []
    if only in {"both", "v0"}:
        scripts += [
            ("v0", root / "slurm_scripts" / "train_from_tuned_all.sh"),
            ("v0", root / "slurm_scripts" / "train_from_tuned_gpu.sh"),
        ]
    if only in {"both", "v2"}:
        scripts += [
            ("v2", root / "slurm_scripts" / "slurm_scripts_v2" / "train_from_tuned_cpu_v2.sh"),
            ("v2", root / "slurm_scripts" / "slurm_scripts_v2" / "train_from_tuned_gpu_v2.sh"),
        ]

    for ver, sp in scripts:
        if not sp.exists():
            continue
        text = _read_text(sp)
        arr = _load_arrays(sp)

        if arr.seeds is not None:
            train_seeds = arr.seeds
        else:
            seed_from_cli = _parse_cli_int(text, "--seed")
            train_seeds = [seed_from_cli if seed_from_cli is not None else v0_seed_default]

        version_tag = _parse_cli_int(text, "--version") if ver == "v2" else None

        tuned_params_literal = _parse_cli_token(text, "--tuned_params_path")
        tuned_params_literal = tuned_params_literal if tuned_params_literal and "$" not in tuned_params_literal else None

        for algorithm in arr.algorithms:
            for set_id in arr.sets:
                for train_seed in train_seeds:
                    if ver == "v0":
                        study_name = _study_name_v0(algorithm, set_id, train_seed)
                    else:
                        study_name = _name_v2(algorithm, set_id, train_seed, version_tag)

                    bestparams_path = (
                        Path(tuned_params_literal)
                        if tuned_params_literal is not None
                        else (root / "logs" / "tuning_logs" / study_name / "best_hyperparameters.yaml")
                    )
                    bestmodel_path = root / "logs" / "tuning_logs" / study_name / "best_model.zip"

                    reuse_seed_suggestion: Optional[Path] = None
                    if ver == "v2" and suggest_reuse_seed is not None and tuned_params_literal is None:
                        if suggest_reuse_seed != train_seed:
                            reuse_study = _name_v2(algorithm, set_id, suggest_reuse_seed, version_tag)
                            reuse_seed_suggestion = root / "logs" / "tuning_logs" / reuse_study / "best_hyperparameters.yaml"

                    yield ExpectedBestparams(
                        version=ver,
                        script=sp,
                        algorithm=algorithm,
                        set_id=set_id,
                        train_seed=train_seed,
                        version_tag=version_tag,
                        study_name=study_name,
                        bestparams_path=bestparams_path,
                        bestmodel_path=bestmodel_path,
                        tuned_params_literal=tuned_params_literal,
                        reuse_seed_suggestion=reuse_seed_suggestion,
                    )


def _check_bestparams_file(p: Path) -> Tuple[bool, str, int]:
    """
    Returns: (ok, reason, size_bytes)
    """
    if not p.exists() or not p.is_file():
        return False, "missing", 0
    try:
        size = p.stat().st_size
    except Exception:
        size = 0
    if size <= 0:
        return False, "empty", size
    if yaml is None:
        return True, "present_no_yaml_validation", size
    obj, err = _read_yaml_if_possible(p)
    if err is not None:
        return False, f"yaml_error:{err}", size
    ok, mode = _yaml_valid_bestparams(obj)
    return ok, mode, size


def _parse_rerun_arrays_for_tuning(root: Path, only: str) -> List[ScriptArrays]:
    scripts: List[Path] = []
    if only in ("v0", "both"):
        scripts += [root / "slurm_scripts" / "tune_all.sh", root / "slurm_scripts" / "tune_gpu.sh"]
    if only in ("v2", "both"):
        scripts += [
            root / "slurm_scripts" / "slurm_scripts_v2" / "tuning_cpu_v2.sh",
            root / "slurm_scripts" / "slurm_scripts_v2" / "tuning_gpu_v2.sh",
        ]
    out: List[ScriptArrays] = []
    for sp in scripts:
        if sp.exists():
            try:
                out.append(_load_arrays(sp))
            except Exception as e:
                print(f"[WARN] Could not parse tuning script {sp}: {e}")
    return out


def _parse_rerun_arrays_for_train_from_tuned(root: Path, only: str) -> List[ScriptArrays]:
    scripts: List[Path] = []
    if only in ("v0", "both"):
        scripts += [root / "slurm_scripts" / "train_from_tuned_all.sh", root / "slurm_scripts" / "train_from_tuned_gpu.sh"]
    if only in ("v2", "both"):
        scripts += [
            root / "slurm_scripts" / "slurm_scripts_v2" / "train_from_tuned_cpu_v2.sh",
            root / "slurm_scripts" / "slurm_scripts_v2" / "train_from_tuned_gpu_v2.sh",
        ]
    out: List[ScriptArrays] = []
    for sp in scripts:
        if sp.exists():
            try:
                out.append(_load_arrays(sp))
            except Exception as e:
                print(f"[WARN] Could not parse train-from-tuned script {sp}: {e}")
    return out


def cmd_before_train_from_tuned(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()

    expected = list(
        _iter_expected_from_train_from_tuned(
            root,
            only=args.only,
            v0_seed_default=args.v0_seed,
            suggest_reuse_seed=args.suggest_reuse_seed,
        )
    )
    if not expected:
        print("ERROR: No train-from-tuned scripts found under expected paths. Check --root / repo layout.", file=sys.stderr)
        return 2

    latest_fail, _scheme_counts = _load_failures_latest(root / args.failed_runs) if args.failures else ({}, Counter())

    # Check tuned YAMLs
    missing: List[ExpectedBestparams] = []
    invalid: List[Tuple[ExpectedBestparams, str]] = []
    ok_count = 0
    restorables: List[Tuple[Path, Path]] = []  # (backup, expected)

    reuse_candidates: List[Tuple[ExpectedBestparams, Path]] = []  # (missing_expected, reuse_path)

    for e in expected:
        ok, reason, _size = _check_bestparams_file(e.bestparams_path)
        if ok:
            ok_count += 1
        else:
            if reason == "missing":
                missing.append(e)
                backup = _find_latest_backup_bestparams(root, e.study_name)
                if backup is not None:
                    restorables.append((backup, e.bestparams_path))
                if e.reuse_seed_suggestion is not None:
                    ok2, _, _ = _check_bestparams_file(e.reuse_seed_suggestion)
                    if ok2:
                        reuse_candidates.append((e, e.reuse_seed_suggestion))
            else:
                invalid.append((e, reason))

    total = len(expected)

    _print_header(f"BEFORE TRAIN-FROM-TUNED  (root={root})  ({_now_local_iso()})")
    print(f"Tuned hyperparameter YAMLs needed: {ok_count}/{total} OK  missing={len(missing)}  invalid={len(invalid)}")
    if yaml is None:
        print("[WARN] PyYAML is not installed; YAML structure cannot be validated (only existence/size).")

    if missing:
        print(f"\nMissing tuned YAMLs (showing up to {args.show}):")
        for e in missing[: args.show]:
            p = e.bestparams_path
            print(f"  - {e.version} {e.study_name}  | expected={_rel(root, p)}")
            backup = _find_latest_backup_bestparams(root, e.study_name)
            if backup:
                print(f"      backup_candidate: {_rel(root, backup)}")
            if e.reuse_seed_suggestion:
                print(f"      reuse_seed_suggestion: {_rel(root, e.reuse_seed_suggestion)}")
            if args.failures:
                fi = latest_fail.get(e.study_name)
                if fi:
                    print(f"      latest_failure: {_fmt_failure_brief(fi)}")

    if invalid:
        print(f"\nInvalid tuned YAMLs (exist but unusable) (showing up to {args.show}):")
        for e, reason in invalid[: args.show]:
            print(f"  - {e.version} {e.study_name}  | {_rel(root, e.bestparams_path)}  | reason={reason}")
            if args.failures:
                fi = latest_fail.get(e.study_name)
                if fi:
                    print(f"      latest_failure: {_fmt_failure_brief(fi)}")

    # Actionable fix: restore from backups
    if restorables:
        print(f"\nRestore options (from newest __old_ backup) (showing up to {args.show}):")
        for src, dst in restorables[: args.show]:
            print(f"  cp -f \"{_rel(root, src)}\" \"{_rel(root, dst)}\"")
        if len(restorables) > args.show:
            print(f"  ... and {len(restorables) - args.show} more")

    # Actionable fix: v2 reuse seed suggestion
    if reuse_candidates:
        print("\nDetected likely v2 seed mismatch: training expects per-seed tuned YAMLs, but another seed's YAML exists.")
        ex, reuse_path = reuse_candidates[0]
        print("Example:")
        print(f"  missing: {_rel(root, ex.bestparams_path)} (train seed={ex.train_seed})")
        print(f"  exists:  {_rel(root, reuse_path)} (reuse seed={args.suggest_reuse_seed})")
        print("\nFix options:")
        print("  1) Tune every seed you plan to train (edit tuning_*_v2.sh seeds=(...) and adjust --array).")
        print("  2) Reuse one seed's tuned YAML for all seeds by adding --tuned_params_path to train_v2.py calls.")
        print(f"     Suggested: --tuned_params_path \"{_rel(root, reuse_path)}\"")

    # Actionable fix: rerun tuning arrays for missing studies (only when missing map cleanly)
    if missing and args.suggest_reruns:
        tuning_arrays = _parse_rerun_arrays_for_tuning(root, args.only)
        if tuning_arrays:
            by_script: Dict[Path, List[int]] = defaultdict(list)
            for e in missing:
                # If this is v2 seed mismatch and the tune scripts don't include that seed, index calc will fail; that's ok.
                for arr in tuning_arrays:
                    if e.algorithm not in arr.algorithms or e.set_id not in arr.sets:
                        continue
                    # Need seed match if script has seeds array
                    seed_for_idx: Optional[int] = e.train_seed if arr.seeds is not None else None
                    if arr.seeds is not None and (seed_for_idx not in arr.seeds):
                        continue
                    try:
                        idx = _compute_array_index(arr, e.algorithm, e.set_id, seed_for_idx)
                    except Exception:
                        continue
                    by_script[arr.path].append(idx)

            if by_script:
                print("\nSuggested sbatch reruns for missing tuned YAMLs (tuning scripts):")
                for sp, idxs in sorted(by_script.items(), key=lambda kv: str(kv[0])):
                    idxs_u = sorted(set(idxs))
                    if idxs_u:
                        print(f"  sbatch --array={','.join(map(str, idxs_u))} {_rel(root, sp)}")
            else:
                print("\n[INFO] No direct sbatch rerun suggestions found (seed/version mismatch or scripts not parsable).")

    # Status of train-from-tuned runs (training_best_logs)
    if args.check_runs:
        print("\nTraining-from-tuned run status (training_best_logs):")
        statuses: List[RunStatus] = []
        for e in expected:
            # train-from-tuned run_name equals study_name in both v0 and v2
            run_name = e.study_name
            run_dir = root / "logs" / "training_best_logs" / run_name
            statuses.append(_summarize_run_dir(run_name, run_dir))

        complete = [s for s in statuses if s.is_present and s.trained_model_ok]
        partial = [s for s in statuses if s.is_present and not s.trained_model_ok]
        missing_dirs = [s for s in statuses if not s.is_present]

        print(f"  expected={len(statuses)}  complete={len(complete)}  partial={len(partial)}  missing_dir={len(missing_dirs)}")

        # Show partial runs with most recent updates first
        partial_sorted = sorted(partial, key=lambda s: s.last_update_ts or 0, reverse=True)
        now = time.time()
        show = args.show
        if partial_sorted:
            print(f"\n  Partial runs (no trained_model.zip yet) (showing up to {show}):")
            for s in partial_sorted[:show]:
                age = _fmt_age_seconds(now - s.last_update_ts) if s.last_update_ts else "-"
                lp = s.last_progress or {}
                print(
                    f"  - {s.run_name} | last_update={_fmt_dt(s.last_update_ts) if s.last_update_ts else '?'} ({age}) "
                    f"| steps={_human_int(lp.get('total_timesteps'))} "
                    f"| ep_rew_mean={_human_int(lp.get('ep_rew_mean'))} "
                    f"| note={s.note or s.last_update_source}"
                )
                if args.failures:
                    fi = latest_fail.get(s.run_name)
                    if fi:
                        print(f"      latest_failure: {_fmt_failure_brief(fi)}")

        if args.show_missing_runs and missing_dirs:
            print(f"\n  Missing run directories (showing up to {show}):")
            for s in missing_dirs[:show]:
                print(f"  - {s.run_name} | expected_dir={_rel(root, s.run_dir)}")

        # Suggest rerun arrays for partial/missing train-from-tuned runs
        if args.suggest_reruns and (partial or missing_dirs):
            train_arrays = _parse_rerun_arrays_for_train_from_tuned(root, args.only)
            by_script2: Dict[Path, List[int]] = defaultdict(list)
            for s in (partial + missing_dirs):
                # Parse algorithm/set/seed from run_name (works for both v0/v2 formats used here)
                m_alg = s.run_name.split("_")[0] if s.run_name else None
                m_set = re.search(r"_set(\d+)_seed(\d+)", s.run_name)
                if not m_alg or not m_set:
                    continue
                set_id = int(m_set.group(1))
                seed = int(m_set.group(2))
                for arr in train_arrays:
                    if m_alg not in arr.algorithms or set_id not in arr.sets:
                        continue
                    seed_for_idx: Optional[int] = seed if arr.seeds is not None else None
                    if arr.seeds is not None and seed_for_idx not in arr.seeds:
                        continue
                    try:
                        idx = _compute_array_index(arr, m_alg, set_id, seed_for_idx)
                    except Exception:
                        continue
                    by_script2[arr.path].append(idx)

            if by_script2:
                print("\n  Suggested sbatch reruns for incomplete train-from-tuned runs:")
                for sp, idxs in sorted(by_script2.items(), key=lambda kv: str(kv[0])):
                    idxs_u = sorted(set(idxs))
                    if idxs_u:
                        print(f"    sbatch --array={','.join(map(str, idxs_u))} {_rel(root, sp)}")

    # Return code: "before" means prerequisites only
    if missing or invalid:
        print("\nRESULT: NOT READY for train-from-tuned (missing/invalid tuned hyperparameters).")
        return 2
    print("\nRESULT: READY for train-from-tuned (tuned hyperparameters present and valid).")
    return 0


# -------------------------
# Command: before_transfer
# -------------------------

@dataclass(frozen=True)
class ExpectedTransfer:
    version: str
    script: Path
    algorithm: str
    load_set: int
    train_set: int
    seed: int
    version_tag: Optional[int]
    source_model_zip: Path
    transfer_run_name: str
    transfer_run_dir: Path


def _iter_expected_transfer(root: Path, *, only: str, v0_seed_default: int, load_set_default: int) -> Iterable[ExpectedTransfer]:
    scripts: List[Tuple[str, Path]] = []
    if only in {"both", "v0"}:
        scripts += [
            ("v0", root / "slurm_scripts" / "transfer_all.sh"),
            ("v0", root / "slurm_scripts" / "transfer_gpu.sh"),
        ]
    if only in {"both", "v2"}:
        scripts += [
            ("v2", root / "slurm_scripts" / "slurm_scripts_v2" / "transfer_cpu_v2.sh"),
            ("v2", root / "slurm_scripts" / "slurm_scripts_v2" / "transfer_gpu_v2.sh"),
        ]

    for ver, sp in scripts:
        if not sp.exists():
            continue
        text = _read_text(sp)
        arr = _load_arrays(sp)

        if arr.seeds is not None:
            seeds = arr.seeds
        else:
            seed_const = _parse_cli_int(text, "--seed") or v0_seed_default
            seeds = [seed_const]

        load_set = _parse_cli_int(text, "--load_set") or load_set_default
        version_tag = _parse_cli_int(text, "--version") if ver == "v2" else None

        train_sets = arr.sets

        for algorithm in arr.algorithms:
            for seed in seeds:
                # Source model path must exist (trained on load_set, default hyperparams)
                if ver == "v0":
                    source_run_name = _study_name_v0(algorithm, load_set, seed)
                else:
                    source_run_name = _name_v2(algorithm, load_set, seed, version_tag)

                source_model_zip = (
                    root
                    / "logs"
                    / "training_default_logs"
                    / source_run_name
                    / "checkpoints"
                    / "trained_model.zip"
                )

                for train_set in train_sets:
                    if ver == "v0":
                        transfer_name = _transfer_name_v0(algorithm, load_set, train_set, seed)
                    else:
                        transfer_name = _transfer_name_v2(algorithm, load_set, train_set, seed, version_tag)

                    transfer_dir = root / "logs" / "transfer_logs" / transfer_name

                    yield ExpectedTransfer(
                        version=ver,
                        script=sp,
                        algorithm=algorithm,
                        load_set=load_set,
                        train_set=train_set,
                        seed=seed,
                        version_tag=version_tag,
                        source_model_zip=source_model_zip,
                        transfer_run_name=transfer_name,
                        transfer_run_dir=transfer_dir,
                    )


def _parse_rerun_arrays_for_training(root: Path, only: str) -> List[ScriptArrays]:
    scripts: List[Path] = []
    if only in ("v0", "both"):
        scripts += [root / "slurm_scripts" / "train_all.sh", root / "slurm_scripts" / "train_gpu.sh"]
    if only in ("v2", "both"):
        scripts += [
            root / "slurm_scripts" / "slurm_scripts_v2" / "training_cpu_v2.sh",
            root / "slurm_scripts" / "slurm_scripts_v2" / "training_gpu_v2.sh",
        ]
    out: List[ScriptArrays] = []
    for sp in scripts:
        if sp.exists():
            try:
                out.append(_load_arrays(sp))
            except Exception as e:
                print(f"[WARN] Could not parse training script {sp}: {e}")
    return out


def _parse_rerun_arrays_for_transfer(root: Path, only: str) -> List[ScriptArrays]:
    scripts: List[Path] = []
    if only in ("v0", "both"):
        scripts += [root / "slurm_scripts" / "transfer_all.sh", root / "slurm_scripts" / "transfer_gpu.sh"]
    if only in ("v2", "both"):
        scripts += [
            root / "slurm_scripts" / "slurm_scripts_v2" / "transfer_cpu_v2.sh",
            root / "slurm_scripts" / "slurm_scripts_v2" / "transfer_gpu_v2.sh",
        ]
    out: List[ScriptArrays] = []
    for sp in scripts:
        if sp.exists():
            try:
                out.append(_load_arrays(sp))
            except Exception as e:
                print(f"[WARN] Could not parse transfer script {sp}: {e}")
    return out


def cmd_before_transfer(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()

    expected = list(_iter_expected_transfer(root, only=args.only, v0_seed_default=args.v0_seed, load_set_default=args.load_set))
    if not expected:
        print("ERROR: No transfer scripts found under expected paths. Check --root / repo layout.", file=sys.stderr)
        return 2

    latest_fail, _scheme_counts = _load_failures_latest(root / args.failed_runs) if args.failures else ({}, Counter())

    # Required source models (unique)
    required_models: Dict[Tuple[str, str, int, int, Optional[int]], Path] = {}
    # key: (ver, alg, seed, load_set, version_tag)
    for e in expected:
        key = (e.version, e.algorithm, e.seed, e.load_set, e.version_tag)
        required_models[key] = e.source_model_zip

    missing: List[Tuple[Tuple[str, str, int, int, Optional[int]], Path]] = []
    bad: List[Tuple[Tuple[str, str, int, int, Optional[int]], Path]] = []
    okc = 0

    for key, p in sorted(required_models.items(), key=lambda kv: str(kv[1])):
        ok, size, iszip = _is_good_zip(p)
        if ok:
            okc += 1
        else:
            if (not p.exists()) or size == 0:
                missing.append((key, p))
            else:
                bad.append((key, p))

    _print_header(f"BEFORE TRANSFER  (root={root})  ({_now_local_iso()})")
    print(f"Required SOURCE models: {okc}/{len(required_models)} OK  missing={len(missing)}  bad={len(bad)}")

    if missing:
        print(f"\nMissing source trained_model.zip files (showing up to {args.show}):")
        for (ver, alg, seed, load_set, vtag), p in missing[: args.show]:
            suffix = "v0" if ver == "v0" else str(vtag) if vtag is not None else "?"
            run_name = f"{alg}_set{load_set}_seed{seed}_{suffix}"
            print(f"  - {ver} {alg} seed{seed} load_set{load_set} | {_rel(root, p)}")
            # Source run progress (training_default_logs)
            run_dir = root / "logs" / "training_default_logs" / run_name
            rs = _summarize_run_dir(run_name, run_dir)
            if rs.is_present:
                now = time.time()
                age = _fmt_age_seconds(now - rs.last_update_ts) if rs.last_update_ts else "-"
                lp = rs.last_progress or {}
                print(
                    f"      source_run_dir: {_rel(root, run_dir)} | last_update={_fmt_dt(rs.last_update_ts) if rs.last_update_ts else '?'} ({age}) "
                    f"| steps={_human_int(lp.get('total_timesteps'))} | ep_rew_mean={_human_int(lp.get('ep_rew_mean'))} | note={rs.note or rs.last_update_source}"
                )
                if args.failures:
                    fi = latest_fail.get(run_name)
                    if fi:
                        print(f"      latest_failure: {_fmt_failure_brief(fi)}")
            else:
                print(f"      source_run_dir: MISSING ({_rel(root, run_dir)})")

    if bad:
        print(f"\nBad source trained_model.zip files (exist but not valid zip) (showing up to {args.show}):")
        for (ver, alg, seed, load_set, vtag), p in bad[: args.show]:
            print(f"  - {ver} {alg} seed{seed} load_set{load_set} | {_rel(root, p)}")

    # Suggest reruns for missing source models
    if (missing or bad) and args.suggest_reruns:
        training_arrays = _parse_rerun_arrays_for_training(root, args.only)
        if training_arrays:
            by_script: Dict[Path, List[int]] = defaultdict(list)
            for (ver, alg, seed, load_set, _vtag), _p in (missing + bad):
                for arr in training_arrays:
                    if alg not in arr.algorithms or load_set not in arr.sets:
                        continue
                    seed_for_idx: Optional[int] = seed if arr.seeds is not None else None
                    if arr.seeds is not None and seed_for_idx not in arr.seeds:
                        continue
                    try:
                        idx = _compute_array_index(arr, alg, load_set, seed_for_idx)
                    except Exception:
                        continue
                    by_script[arr.path].append(idx)

            if by_script:
                print("\nSuggested sbatch reruns for missing SOURCE models (training scripts):")
                for sp, idxs in sorted(by_script.items(), key=lambda kv: str(kv[0])):
                    idxs_u = sorted(set(idxs))
                    if idxs_u:
                        print(f"  sbatch --array={','.join(map(str, idxs_u))} {_rel(root, sp)}")

    # Transfer run status (transfer_logs)
    if args.check_runs:
        print("\nTransfer run status (transfer_logs):")
        run_names = sorted(set(e.transfer_run_name for e in expected))
        statuses: List[RunStatus] = []
        for rn in run_names:
            rd = root / "logs" / "transfer_logs" / rn
            statuses.append(_summarize_run_dir(rn, rd))

        complete = [s for s in statuses if s.is_present and s.trained_model_ok]
        partial = [s for s in statuses if s.is_present and not s.trained_model_ok]
        missing_dirs = [s for s in statuses if not s.is_present]

        print(f"  expected={len(statuses)}  complete={len(complete)}  partial={len(partial)}  missing_dir={len(missing_dirs)}")

        now = time.time()
        show = args.show
        if partial:
            partial_sorted = sorted(partial, key=lambda s: s.last_update_ts or 0, reverse=True)
            print(f"\n  Partial transfer runs (showing up to {show}):")
            for s in partial_sorted[:show]:
                age = _fmt_age_seconds(now - s.last_update_ts) if s.last_update_ts else "-"
                lp = s.last_progress or {}
                print(
                    f"  - {s.run_name} | last_update={_fmt_dt(s.last_update_ts) if s.last_update_ts else '?'} ({age}) "
                    f"| steps={_human_int(lp.get('total_timesteps'))} | ep_rew_mean={_human_int(lp.get('ep_rew_mean'))} | note={s.note or s.last_update_source}"
                )
                if args.failures:
                    fi = latest_fail.get(s.run_name)
                    if fi:
                        print(f"      latest_failure: {_fmt_failure_brief(fi)}")

        if args.show_missing_runs and missing_dirs:
            print(f"\n  Missing transfer run directories (showing up to {show}):")
            for s in missing_dirs[:show]:
                print(f"  - {s.run_name} | expected_dir={_rel(root, s.run_dir)}")

        # v0 warning about reruns
        if args.only in ("v0", "both"):
            v0_existing = [s for s in statuses if s.run_name.endswith("_v0") and s.is_present]
            if v0_existing:
                print("\n[WARN] v0 transfer reruns can mix logs (v0 transfer.py does NOT backup).")
                print("      If you need a clean rerun, move/delete the existing run directory first.")

        # Suggest reruns for partial/missing transfer runs
        if args.suggest_reruns and (partial or missing_dirs):
            transfer_arrays = _parse_rerun_arrays_for_transfer(root, args.only)
            by_script2: Dict[Path, List[int]] = defaultdict(list)
            for s in (partial + missing_dirs):
                # Parse algorithm/from/to/seed from run_name
                m_alg = s.run_name.split("_")[0] if s.run_name else None
                m = re.search(r"_from(\d+)_to(\d+)_seed(\d+)", s.run_name)
                if not m_alg or not m:
                    continue
                train_set = int(m.group(2))
                seed = int(m.group(3))
                for arr in transfer_arrays:
                    if m_alg not in arr.algorithms or train_set not in arr.sets:
                        continue
                    seed_for_idx: Optional[int] = seed if arr.seeds is not None else None
                    if arr.seeds is not None and seed_for_idx not in arr.seeds:
                        continue
                    try:
                        idx = _compute_array_index(arr, m_alg, train_set, seed_for_idx)
                    except Exception:
                        continue
                    by_script2[arr.path].append(idx)

            if by_script2:
                print("\n  Suggested sbatch reruns for incomplete transfer runs:")
                for sp, idxs in sorted(by_script2.items(), key=lambda kv: str(kv[0])):
                    idxs_u = sorted(set(idxs))
                    if idxs_u:
                        print(f"    sbatch --array={','.join(map(str, idxs_u))} {_rel(root, sp)}")

    # Return code: "before" checks prerequisites only
    if missing or bad:
        print("\nRESULT: NOT READY for transfer (missing/bad SOURCE models).")
        return 2
    print("\nRESULT: READY for transfer (all required SOURCE models exist).")
    return 0


# -------------------------
# Command: globs
# -------------------------

def _run_dir_from_event_path(event_path: str) -> str:
    """
    Extract the run directory name from an event-file path.

    Works for both relative paths like:
        logs/training_default_logs/<RUN>/tensorboard/.../events.out.tfevents...
    and absolute paths where `logs/` appears somewhere in the middle.
    """
    parts = event_path.split(os.sep)
    try:
        ix = parts.index("logs")
        # logs/<log_type>/<run_dir>/...
        if ix + 2 < len(parts):
            return parts[ix + 2]
    except ValueError:
        pass
    # Fallback: best-effort (old behavior) for already-relative paths
    return parts[2] if len(parts) > 2 else ""


def _trial_from_event_path(event_path: str) -> Optional[int]:
    parts = event_path.split(os.sep)
    if "trials" not in parts:
        return None
    try:
        tix = parts.index("trials")
        if tix + 1 >= len(parts):
            return None
        m = re.match(r"trial_(\d+)", parts[tix + 1])
        return int(m.group(1)) if m else None
    except ValueError:
        return None


def _range_str(values: Set[int], *, max_list: int = 6) -> str:
    if not values:
        return "-"
    vals = sorted(values)
    if len(vals) == 1:
        return str(vals[0])
    if len(vals) <= max_list:
        return "[" + ",".join(map(str, vals)) + "]"
    return f"{vals[0]}-{vals[-1]} (n={len(vals)})"


def _algo_str(algo_counts: Counter, *, show_counts: bool) -> str:
    if not algo_counts:
        return "0"
    if show_counts:
        bits = ",".join([f"{a}={c}" for a, c in sorted(algo_counts.items())])
        return f"{len(algo_counts)}({bits})"
    names = ",".join(sorted(algo_counts.keys()))
    return f"{len(algo_counts)}({names})"


def _summarize_glob(pattern: str) -> Dict[str, Any]:
    event_files = glob.glob(pattern, recursive=True)
    run_dirs: Set[str] = set()
    trials: Set[int] = set()

    for p in event_files:
        rd = _run_dir_from_event_path(p)
        if rd:
            run_dirs.add(rd)
        t = _trial_from_event_path(p)
        if t is not None:
            trials.add(t)

    algo_counts = Counter()
    seeds: Set[int] = set()
    sets_: Set[int] = set()
    from_sets: Set[int] = set()
    to_sets: Set[int] = set()

    for r in run_dirs:
        algo = r.split("_")[0] if r else ""
        if algo:
            algo_counts[algo] += 1

        m = re.search(r"_seed(\d+)", r)
        if m:
            seeds.add(int(m.group(1)))

        m = re.search(r"_set(\d+)", r)
        if m:
            sets_.add(int(m.group(1)))

        m = re.search(r"_from(\d+)_to(\d+)", r)
        if m:
            from_sets.add(int(m.group(1)))
            to_sets.add(int(m.group(2)))

    return {
        "pattern": pattern,
        "event_files": len(event_files),
        "runs": len(run_dirs),
        "algo_counts": algo_counts,
        "seeds": seeds,
        "sets": sets_,
        "from_sets": from_sets,
        "to_sets": to_sets,
        "trials": trials,
    }


def _print_glob_line(tag: str, s: Dict[str, Any], *, show_patterns: bool, show_algo_counts: bool) -> None:
    algos = _algo_str(s["algo_counts"], show_counts=show_algo_counts)
    seeds = _range_str(s["seeds"])
    sets_ = _range_str(s["sets"])
    from_sets = _range_str(s["from_sets"])
    to_sets = _range_str(s["to_sets"])
    trials = _range_str(s["trials"])

    line = (
        f"{tag}: events={s['event_files']} runs={s['runs']} "
        f"algos={algos} seeds={seeds} sets={sets_} from={from_sets} to={to_sets} trials={trials}"
    )
    if show_patterns:
        line += f" | glob={s['pattern']}"
    print(line)


def cmd_globs(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    show_patterns = not args.no_patterns
    show_algo_counts = bool(args.algo_counts)

    v0_run_glob = f"*_seed{args.v0_seed}_v0"
    v2_run_glob = f"*_seed{args.v2_seed}_2" if args.v2_seed is not None else "*_2"

    def pat(rel_pattern: str) -> str:
        # glob.glob wants a string; rel_pattern is relative to repo root.
        return str(root / rel_pattern)

    _print_header(f"GLOBS / PLOTTING PICKUP  (root={root})  ({_now_local_iso()})")
    print("This shows what your plotting/table scripts will pick up (via event file globs).")

    print(f"\n# generate_table.py (v0) RUN_GLOB={v0_run_glob}")
    _print_glob_line(
        "TABLE v0 A",
        _summarize_glob(pat(f"logs/training_default_logs/{v0_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "TABLE v0 B",
        _summarize_glob(pat(f"logs/training_best_logs/{v0_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "TABLE v0 C",
        _summarize_glob(pat(f"logs/transfer_logs/{v0_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )

    print(f"\n# generate_table_v2.py (v2) RUN_GLOB={v2_run_glob}")
    _print_glob_line(
        "TABLE v2 A",
        _summarize_glob(pat(f"logs/training_default_logs/{v2_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "TABLE v2 B",
        _summarize_glob(pat(f"logs/training_best_logs/{v2_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "TABLE v2 C",
        _summarize_glob(pat(f"logs/transfer_logs/{v2_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )

    print("\n# plot_results.py (v0)")
    _print_glob_line(
        "PLOT v0 A",
        _summarize_glob(pat("logs/training_default_logs/*_v0/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "PLOT v0 B",
        _summarize_glob(pat("logs/training_best_logs/*_v0/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "PLOT v0 C",
        _summarize_glob(pat("logs/transfer_logs/*_v0/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "PLOT v0 Optuna",
        _summarize_glob(pat("logs/tuning_logs/*_v0/trials/trial_*/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )

    print(f"\n# plot_results_v2.py (v2) RUN_GLOB={v2_run_glob}")
    _print_glob_line(
        "PLOT v2 A",
        _summarize_glob(pat(f"logs/training_default_logs/{v2_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "PLOT v2 B",
        _summarize_glob(pat(f"logs/training_best_logs/{v2_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "PLOT v2 C",
        _summarize_glob(pat(f"logs/transfer_logs/{v2_run_glob}/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )
    _print_glob_line(
        "PLOT v2 Optuna",
        _summarize_glob(pat(f"logs/tuning_logs/{v2_run_glob}/trials/trial_*/tensorboard/**/events.out.tfevents.*")),
        show_patterns=show_patterns,
        show_algo_counts=show_algo_counts,
    )

    return 0



# -------------------------
# Command: failures
# -------------------------

def cmd_failures(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    failed_runs = root / args.failed_runs
    if not failed_runs.exists():
        print(f"No failed runs file found: {failed_runs}")
        return 0

    # Stream parse, optionally filter, keep tail
    tail: Deque[FailureInfo] = deque(maxlen=args.tail)
    counts_scheme: Counter = Counter()
    counts_exc: Counter = Counter()
    counts_script: Counter = Counter()

    def _match(fi: FailureInfo) -> bool:
        if args.scheme and fi.scheme != args.scheme:
            return False
        if args.run_name and (not fi.run_name or args.run_name not in fi.run_name):
            return False
        if args.script and args.script not in fi.script:
            return False
        return True

    with failed_runs.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fi = FailureInfo.from_json(obj)
            if not _match(fi):
                continue
            counts_scheme[fi.scheme] += 1
            counts_script[fi.script] += 1
            if fi.exc_type:
                counts_exc[fi.exc_type] += 1
            tail.append(fi)

    _print_header(f"FAILURES SUMMARY  (root={root})  ({_now_local_iso()})")
    print(f"File: {_rel(root, failed_runs)}")
    if args.scheme:
        print(f"Filter: scheme={args.scheme}")
    if args.run_name:
        print(f"Filter: run_name contains '{args.run_name}'")
    if args.script:
        print(f"Filter: script contains '{args.script}'")

    total = sum(counts_scheme.values())
    print(f"\nMatching failures: {total}")
    if total == 0:
        return 0

    print("\nBy scheme:")
    for k, v in counts_scheme.most_common():
        print(f"  {k}: {v}")

    print("\nBy script:")
    for k, v in counts_script.most_common():
        print(f"  {k}: {v}")

    print("\nTop exception types:")
    for k, v in counts_exc.most_common(12):
        print(f"  {k}: {v}")

    print(f"\nMost recent {len(tail)} failures:")
    for fi in list(tail):
        rn = fi.run_name or "-"
        print(f"- {fi.ts} | {fi.scheme} | {fi.script} | {rn}")
        msg = _truncate(str(fi.exc or ""), 200) if fi.exc else ""
        if fi.exc_type or msg:
            print(f"    {fi.exc_type or ''} {msg}".strip())
        # Slurm IDs are often the most actionable
        if fi.slurm:
            aj = fi.slurm.get("SLURM_ARRAY_JOB_ID") or fi.slurm.get("SLURM_JOB_ID")
            at = fi.slurm.get("SLURM_ARRAY_TASK_ID")
            if aj or at:
                print(f"    slurm: job={aj} task={at}")

    return 0


# -------------------------
# Command: sync_failures (slurm scan)
# -------------------------

LOGFILE_RE = re.compile(r"^(?P<jobname>.+)_(?P<jobid>\d+)_(?P<taskid>\d+)\.(?P<ext>out|err)$")


@dataclass(frozen=True)
class SlurmLogKey:
    job_name: str
    array_job_id: str
    array_task_id: str
    directory: str


@dataclass
class SlurmLogPair:
    key: SlurmLogKey
    out_path: Optional[Path]
    err_path: Optional[Path]


@dataclass
class FailureSignal:
    is_failure: bool
    exc_type: Optional[str] = None
    exc_msg: Optional[str] = None
    traceback_text: Optional[str] = None
    reason: Optional[str] = None


def _read_text_head_tail(path: Path, *, head_bytes: int = 64_000, tail_bytes: int = 128_000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
    except Exception:
        size = None
    try:
        with path.open("rb") as f:
            head = f.read(head_bytes)
            if size is None:
                return head.decode("utf-8", errors="replace")
            if size <= head_bytes + tail_bytes:
                f.seek(0)
                data = f.read()
                return data.decode("utf-8", errors="replace")
            f.seek(-tail_bytes, os.SEEK_END)
            tail = f.read(tail_bytes)
        return head.decode("utf-8", errors="replace") + "\n\n... [truncated] ...\n\n" + tail.decode("utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR reading {path}: {e}]"


def _detect_failure(out_text: str, err_text: str) -> FailureSignal:
    text = (err_text or "") + "\n" + (out_text or "")
    lowered = text.lower()

    tb_marker = "traceback (most recent call last):"
    if tb_marker in lowered:
        idx = lowered.rfind(tb_marker)
        tb = text[idx:]
        last_line = ""
        for line in reversed(tb.splitlines()):
            if line.strip():
                last_line = line.strip()
                break
        exc_type = None
        exc_msg = None
        m = re.match(r"^([A-Za-z_][\w\.]*)(?::\s*(.*))?$", last_line)
        if m:
            exc_type = m.group(1)
            exc_msg = m.group(2)
        return FailureSignal(
            is_failure=True,
            exc_type=exc_type or "PythonException",
            exc_msg=exc_msg or last_line,
            traceback_text=tb.strip()[:200_000],
            reason="python_traceback",
        )

    slurm_patterns: List[Tuple[str, str]] = [
        ("oom-kill", "slurm_oom_kill"),
        ("cuda out of memory", "cuda_out_of_memory"),
        ("out of memory", "out_of_memory"),
        ("due to time limit", "slurm_time_limit"),
        ("time limit", "slurm_time_limit"),
        ("cancelled", "slurm_cancelled"),
        ("slurmstepd: error", "slurmstepd_error"),
        ("srun: error", "srun_error"),
        ("segmentation fault", "segfault"),
        ("core dumped", "core_dump"),
    ]
    for needle, reason in slurm_patterns:
        if needle in lowered:
            pos = lowered.find(needle)
            lo = max(pos - 400, 0)
            hi = min(pos + 800, len(text))
            excerpt = text[lo:hi].strip()
            return FailureSignal(
                is_failure=True,
                exc_type="SlurmOrSystemError",
                exc_msg=excerpt[:4000],
                traceback_text=None,
                reason=reason,
            )

    shell_patterns: List[Tuple[str, str]] = [
        ("command not found", "command_not_found"),
        ("no such file or directory", "no_such_file"),
        ("conda: error", "conda_error"),
        ("exited with exit code", "exit_code"),
    ]
    for needle, reason in shell_patterns:
        if needle in lowered:
            pos = lowered.find(needle)
            lo = max(pos - 400, 0)
            hi = min(pos + 800, len(text))
            excerpt = text[lo:hi].strip()
            return FailureSignal(
                is_failure=True,
                exc_type="ShellError",
                exc_msg=excerpt[:4000],
                traceback_text=None,
                reason=reason,
            )

    return FailureSignal(is_failure=False)


def _iter_slurm_log_pairs(output_dirs: Sequence[Path]) -> Iterable[SlurmLogPair]:
    by_key: Dict[SlurmLogKey, SlurmLogPair] = {}
    for d in output_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            m = LOGFILE_RE.match(p.name)
            if not m:
                continue
            key = SlurmLogKey(
                job_name=m.group("jobname"),
                array_job_id=m.group("jobid"),
                array_task_id=m.group("taskid"),
                directory=str(d),
            )
            pair = by_key.get(key)
            if pair is None:
                pair = SlurmLogPair(key=key, out_path=None, err_path=None)
                by_key[key] = pair
            if m.group("ext") == "out":
                pair.out_path = p
            else:
                pair.err_path = p
    yield from by_key.values()


def _load_existing_failure_keys(failed_runs_path: Path) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    if not failed_runs_path.exists():
        return keys
    with failed_runs_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sl = obj.get("slurm") or {}
            aj = sl.get("SLURM_ARRAY_JOB_ID") or sl.get("SLURM_JOB_ID")
            at = sl.get("SLURM_ARRAY_TASK_ID")
            if aj and at:
                keys.add((str(aj), str(at)))
    return keys


def _append_failures_locked(failed_runs_path: Path, entries: List[Dict[str, Any]]) -> int:
    if not entries:
        return 0
    if fcntl is None:
        raise RuntimeError("fcntl is not available; cannot lock failed_runs.jsonl safely on this platform")
    failed_runs_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with failed_runs_path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            existing = _load_existing_failure_keys_from_handle(f)
            f.seek(0, os.SEEK_END)
            for entry in entries:
                sl = entry.get("slurm") or {}
                aj = sl.get("SLURM_ARRAY_JOB_ID") or sl.get("SLURM_JOB_ID")
                at = sl.get("SLURM_ARRAY_TASK_ID")
                if aj and at and (str(aj), str(at)) in existing:
                    continue
                f.write(json.dumps(entry) + "\n")
                appended += 1
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return appended


def _load_existing_failure_keys_from_handle(f) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    f.seek(0)
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        sl = obj.get("slurm") or {}
        aj = sl.get("SLURM_ARRAY_JOB_ID") or sl.get("SLURM_JOB_ID")
        at = sl.get("SLURM_ARRAY_TASK_ID")
        if aj and at:
            keys.add((str(aj), str(at)))
    return keys


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def cmd_sync_failures(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    failed_runs_path = root / args.failed_runs
    output_dirs = [root / d for d in args.dirs]

    pairs = list(_iter_slurm_log_pairs(output_dirs))
    if not pairs:
        print("No Slurm logs found in:")
        for d in output_dirs:
            print(f"  - {d}")
        return 0

    cutoff = time.time() - args.min_mtime_age_sec
    existing_keys = _load_existing_failure_keys(failed_runs_path)

    candidates: List[Dict[str, Any]] = []
    scanned = 0
    failures_found = 0

    for pair in pairs:
        if (pair.key.array_job_id, pair.key.array_task_id) in existing_keys:
            continue

        newest_mtime = 0.0
        for p in [pair.out_path, pair.err_path]:
            if p and p.exists():
                try:
                    newest_mtime = max(newest_mtime, p.stat().st_mtime)
                except Exception:
                    pass
        if newest_mtime and newest_mtime > cutoff:
            continue

        out_text = _read_text_head_tail(pair.out_path) if pair.out_path else ""
        err_text = _read_text_head_tail(pair.err_path) if pair.err_path else ""
        scanned += 1

        sig = _detect_failure(out_text, err_text)
        if not sig.is_failure:
            continue

        failures_found += 1
        entry: Dict[str, Any] = {
            "ts": _now_iso(),
            "host": socket.gethostname(),
            "scheme": "slurm_scan",
            "script": "check_status.py",
            "run_name": None,
            "argv": ["check_status.py", "sync_failures"],
            "slurm": {
                "SLURM_JOB_NAME": pair.key.job_name,
                "SLURM_ARRAY_JOB_ID": pair.key.array_job_id,
                "SLURM_ARRAY_TASK_ID": pair.key.array_task_id,
            },
            "exc_type": sig.exc_type,
            "exc": sig.exc_msg,
            "traceback": sig.traceback_text,
            "reason": sig.reason,
            "slurm_out_path": str(pair.out_path) if pair.out_path else None,
            "slurm_err_path": str(pair.err_path) if pair.err_path else None,
        }
        candidates.append(entry)

    appended = 0
    if args.write:
        try:
            appended = _append_failures_locked(failed_runs_path, candidates)
        except Exception as e:
            print(f"ERROR: failed to append: {e}", file=sys.stderr)
            return 2

    _print_header(f"SYNC FAILURES FROM SLURM LOGS  (root={root})  ({_now_local_iso()})")
    print("Scanned log dirs:")
    for d in output_dirs:
        print(f"  - {_rel(root, d)}")
    print(f"\nLog pairs discovered: {len(pairs)}")
    print(f"Log pairs inspected (old enough & not already recorded): {scanned}")
    print(f"Failures detected: {failures_found}")

    if args.write:
        print(f"Appended {appended} new entries to {_rel(root, failed_runs_path)}")
    else:
        print(f"Dry run: would append {len(candidates)} new entries to {_rel(root, failed_runs_path)}")
        print("Use --write to append.")

    if args.print_new and candidates:
        print("\nNew failure entries:")
        for c in candidates[: args.show]:
            sl = c.get("slurm") or {}
            print(f"- job={sl.get('SLURM_ARRAY_JOB_ID')} task={sl.get('SLURM_ARRAY_TASK_ID')} name={sl.get('SLURM_JOB_NAME')}")
            print(f"  reason={c.get('reason')} exc_type={c.get('exc_type')}")
            print(f"  exc={_truncate(str(c.get('exc') or ''), 220)}")

    return 0


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    # Put common flags in a parent parser so they work both:
    #   python check_status.py --root . tuning ...
    # and
    #   python check_status.py tuning --root . ...
    #
    # IMPORTANT: use default=argparse.SUPPRESS so subparsers don't overwrite
    # values set by the main parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, help="Repo root (where logs/ and slurm_scripts/ live)")
    common.add_argument("--failed_runs", default=argparse.SUPPRESS, help="failed_runs jsonl (relative to --root)")

    ap = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)

    # tuning
    p = sub.add_parser("tuning", help="Check tuning completeness and show what's missing", parents=[common])
    p.add_argument("--tuning_dir", default="logs/tuning_logs", help="Relative path to tuning logs dir")
    p.add_argument("--version", choices=["all", "v0", "v2"], default="all")
    p.add_argument("--expected", type=int, default=20, help="Fallback expected trials if study_config.yaml missing")
    p.add_argument("--require_events", action="store_true", help="Require each trial dir to have at least one event file")
    # p.add_argument("--no-require-events", dest="require_events", action="store_false", help="Do NOT require each trial dir to have at least one event file",)
    # p.set_defaults(require_events=True)
    p.add_argument("--require_bestmodel", action="store_true", help="Require best_model.zip to exist and be a valid zip")
    # p.add_argument("--no-require-bestmodel", dest="require_bestmodel", action="store_false", help="Do NOT require best_model.zip to exist and be a valid zip",)
    # p.set_defaults(require_bestmodel=True)
    p.add_argument("--show", type=int, default=25, help="How many items to list in detail")
    p.add_argument("--failures", action="store_true", help="Show latest failure per study (from failed_runs.jsonl)")
    p.set_defaults(func=cmd_tuning)

    # before_train_from_tuned
    p = sub.add_parser("before_train_from_tuned", help="Preflight for train-from-tuned: tuned YAMLs + run progress", parents=[common])
    p.add_argument("--only", choices=["v0", "v2", "both"], default="both")
    p.add_argument("--v0_seed", type=int, default=33, help="Fallback seed for v0 scripts if --seed cannot be parsed")
    p.add_argument("--suggest_reuse_seed", type=int, default=0, help="For v2: suggest reusing this seed's tuned YAML if others are missing")
    p.add_argument("--show", type=int, default=25, help="How many items to list in detail")
    p.add_argument("--check_runs", action="store_true", help="Also summarize training_best_logs run status")
    p.add_argument("--show_missing_runs", action="store_true", help="List missing run directories (in addition to partial runs)")
    p.add_argument("--suggest_reruns", action="store_true", help="Print sbatch --array suggestions for reruns")
    p.add_argument("--failures", action="store_true", help="Show latest failure per run/study (from failed_runs.jsonl)")
    p.set_defaults(func=cmd_before_train_from_tuned)

    # before_transfer
    p = sub.add_parser("before_transfer", help="Preflight for transfer: source models + transfer run progress", parents=[common])
    p.add_argument("--only", choices=["v0", "v2", "both"], default="both")
    p.add_argument("--v0_seed", type=int, default=33, help="Fallback seed for v0 scripts if --seed cannot be parsed")
    p.add_argument("--load_set", type=int, default=1, help="Transfer load_set (default 1)")
    p.add_argument("--show", type=int, default=25, help="How many items to list in detail")
    p.add_argument("--check_runs", action="store_true", help="Also summarize transfer_logs run status")
    p.add_argument("--show_missing_runs", action="store_true", help="List missing transfer run directories (in addition to partial runs)")
    p.add_argument("--suggest_reruns", action="store_true", help="Print sbatch --array suggestions for reruns")
    p.add_argument("--failures", action="store_true", help="Show latest failure per run (from failed_runs.jsonl)")
    p.set_defaults(func=cmd_before_transfer)

    # globs
    p = sub.add_parser("globs", help="Show what plotting/table scripts will pick up via event-file globs", parents=[common])
    p.add_argument("--v0-seed", type=int, default=33, dest="v0_seed", help="Seed used by v0 scripts (default 33)")
    p.add_argument("--v2-seed", type=int, default=None, dest="v2_seed", help="Filter for v2 seeds (0/1). Omit for all seeds.")
    p.add_argument("--no-patterns", action="store_true", help="Hide the raw glob patterns (shorter output)")
    p.add_argument("--algo-counts", action="store_true", help="Show per-algorithm run-dir counts (PPO=10) instead of listing algos")
    p.set_defaults(func=cmd_globs)

    # failures
    p = sub.add_parser("failures", help="Summarize failed_runs.jsonl", parents=[common])
    p.add_argument("--tail", type=int, default=25, help="Show the most recent N failures")
    p.add_argument("--scheme", default=None, help="Filter by scheme (e.g., tune, tune_trial, train, transfer, slurm_scan)")
    p.add_argument("--run-name", dest="run_name", default=None, help="Filter: run_name contains this substring")
    p.add_argument("--script", default=None, help="Filter: script contains this substring")
    p.set_defaults(func=cmd_failures)

    # sync_failures
    p = sub.add_parser("sync_failures", help="Scan Slurm out/err logs and (optionally) append new failures", parents=[common])
    p.add_argument(
        "--dirs",
        nargs="*",
        default=["slurm_scripts/slurm_out", "slurm_scripts/slurm_scripts_v2/out"],
        help="One or more Slurm output directories (relative to --root)",
    )
    p.add_argument(
        "--min_mtime_age_sec",
        type=int,
        default=300,
        help="Only consider log files whose mtime is at least this many seconds old",
    )
    p.add_argument("--write", action="store_true", help="Append NEW failures to failed_runs.jsonl (default is dry-run)")
    p.add_argument("--print_new", action="store_true", help="Print newly detected failures (useful with dry-run)")
    p.add_argument("--show", type=int, default=25, help="Max new entries to print")
    p.set_defaults(func=cmd_sync_failures)

    return ap



def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)

    # Fill defaults if the option wasn't provided in either the main parser
    # or the subparser (because we used default=SUPPRESS).
    if not hasattr(args, "root"):
        args.root = "."
    if not hasattr(args, "failed_runs"):
        args.failed_runs = "failed_runs.jsonl"

    return int(args.func(args))



if __name__ == "__main__":
    raise SystemExit(main())
