#!/usr/bin/env python3
"""
analyze_results.py — Aggregate per-run evaluations.npz files (written by
SB3's EvalCallback during training) into table-ready summary CSVs that
directly map to the LaTeX tables in writings/0_main.tex.

All input CSVs (produced by evaluate.py) and all output CSVs / LaTeX
files are robot-tagged (uav vs wheeled) so the two robot platforms
never overwrite each other's results.

Outputs (written to --results_dir, default "logs/results/"):
  main_default_summary_<robot>.csv     / main_tuned_summary_<robot>.csv
  main_combined_latex_rows_<robot>.txt -> tab:main_results (per-robot half)
  main_combined_latex_rows_both.txt    -> tab:main_results (UAV + Wheeled merged)
  ablation_reward_agg_<robot>.csv      -> tab:ablation_reward
  ablation_obs_agg_<robot>.csv         -> tab:ablation_obs
  ablation_uncertainty_agg_<robot>.csv -> tab:ablation_uncertainty
  dr_results_agg_<robot>.csv           -> tab:dr_results

Inputs expected for ablation_reward, ablation_uncertainty, dr:
  <results_dir>/ablation_reward_<robot>.csv
  <results_dir>/ablation_uncertainty_<robot>.csv
  <results_dir>/dr_inDist_<robot>.csv
  <results_dir>/dr_OOD_<robot>.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ranksums

# ================================================================
# Experiment metadata (robot-type dependent)
# ================================================================

ALGORITHMS = ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"]
SEEDS      = [42, 123, 9999]

# Per-robot env sets and num_robots columns
ENV_SETS_BY_ROBOT = {
    "uav":     list(range(1, 11)),
    "wheeled": list(range(1, 11)),
}
ROBOT_COUNTS_BY_ROBOT = {
    "uav":     [2, 3, 4, 5],
    "wheeled": [2, 3, 4, 5],
}


def _env_sets(robot_type):
    return ENV_SETS_BY_ROBOT[robot_type]


def _robot_counts(robot_type):
    return ROBOT_COUNTS_BY_ROBOT[robot_type]


# ================================================================
# NPZ loading helpers
# ================================================================

def load_npz_best_reward(npz_path: str) -> float:
    data    = np.load(npz_path)
    results = data["results"]
    means   = results.mean(axis=1)
    return float(np.max(means))


def load_npz_ep_length_at_best(npz_path: str) -> float:
    data     = np.load(npz_path)
    results  = data["results"]
    best_idx = int(np.argmax(results.mean(axis=1)))
    if "ep_lengths" in data:
        return float(data["ep_lengths"][best_idx].mean())
    return float("nan")


def find_npz(log_root: str, version: str, tag: str) -> str | None:
    path = os.path.join(log_root, version, tag, "eval_logs", "evaluations.npz")
    return path if os.path.exists(path) else None


def collect_rewards(log_root: str, version: str,
                    alg_list, robot_list, set_list, seed_list,
                    robot_type: str = "uav",
                    extra_fields: dict | None = None) -> list[dict]:
    """Scan NPZ files and return one dict per run."""
    rows    = []
    missing = 0
    for alg in alg_list:
        for N in robot_list:
            for s in set_list:
                for seed in seed_list:
                    tag      = f"{alg}_{robot_type}_N{N}_env{s}_seed{seed}"
                    npz_path = find_npz(log_root, version, tag)
                    if npz_path is None:
                        missing += 1
                        continue
                    r      = load_npz_best_reward(npz_path)
                    ep_len = load_npz_ep_length_at_best(npz_path)
                    row = dict(algorithm=alg, num_robots=N, env_set=s, seed=seed,
                               mean_reward=r, mean_ep_length=ep_len)
                    if extra_fields:
                        row.update(extra_fields)
                    rows.append(row)
    if missing:
        print(f"    [INFO] {missing} NPZ file(s) not found under {log_root}/{version}/")
    return rows


# ================================================================
# Statistical helpers
# ================================================================

def compute_iqm(vals: np.ndarray) -> float:
    q25, q75 = np.percentile(vals, [25, 75])
    mask = (vals >= q25) & (vals <= q75)
    return float(np.mean(vals[mask])) if mask.any() else float(np.mean(vals))


def cvar_0_1(vals: np.ndarray) -> float:
    n = max(1, int(np.ceil(0.1 * len(vals))))
    return float(np.mean(np.sort(vals)[:n]))


def mannwhitney_pval(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U test (Wilcoxon rank-sum) for independent samples."""
    if len(a) < 2 or len(b) < 2: return 1.0
    _, p = ranksums(a, b)
    return float(p)


def mark_best(df_summary: pd.DataFrame, value_col: str = "mean_reward",
              group_cols: list = None, alpha: float = 0.05) -> pd.DataFrame:
    df_summary = df_summary.copy()
    df_summary["is_best"] = False
    df_iter = (df_summary.groupby(group_cols) if group_cols else [(None, df_summary)])

    for _, grp in df_iter:
        sorted_grp = grp.sort_values(value_col, ascending=False)
        if len(sorted_grp) < 2:
            df_summary.loc[sorted_grp.index[0], "is_best"] = True
            continue
        best_idx   = sorted_grp.index[0]
        second_idx = sorted_grp.index[1]
        a = np.array(grp.loc[best_idx,   "raw_rewards"])
        b = np.array(grp.loc[second_idx, "raw_rewards"])
        p = mannwhitney_pval(a, b)
        df_summary.loc[best_idx, "is_best"] = (p < alpha)

    return df_summary


# ================================================================
# Main results (default / tuned HPs)
# ================================================================

def process_main(log_root: str, results_dir: str, hp_tag: str,
                 robot_type: str = "uav") -> pd.DataFrame:
    version = f"main_{hp_tag}"
    print(f"  Scanning {log_root}/{version}/  (robot_type={robot_type})")

    robot_counts = _robot_counts(robot_type)
    env_sets     = _env_sets(robot_type)

    rows = collect_rewards(log_root, version, ALGORITHMS, robot_counts,
                           env_sets, SEEDS, robot_type=robot_type)
    if not rows:
        print(f"  [WARN] No data found for main_{hp_tag} ({robot_type}) — skipping.")
        return None

    df = pd.DataFrame(rows)
    summary_rows = []
    for (alg, N), grp in df.groupby(["algorithm", "num_robots"]):
        r  = grp["mean_reward"].astype(float).values
        el = grp["mean_ep_length"].astype(float).values
        summary_rows.append(dict(
            algorithm=alg, num_robots=N, mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)), max_reward=float(np.max(r)),
            iqm=compute_iqm(r), raw_rewards=list(r), n_runs=len(r),
            mean_ep_length=float(np.nanmean(el)),
        ))

    summary = pd.DataFrame(summary_rows)
    summary = mark_best(summary, "mean_reward", group_cols=["num_robots"])

    out = summary.drop(columns=["raw_rewards"])
    out_path = os.path.join(results_dir, f"main_{hp_tag}_summary_{robot_type}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")

    _write_latex_main(summary, hp_tag, results_dir, robot_type)
    return summary


def _write_latex_main(summary: pd.DataFrame, hp_tag: str,
                      results_dir: str, robot_type: str):
    robot_counts = _robot_counts(robot_type)
    lines = [
        f"% LaTeX table rows for tab:{'default' if hp_tag == 'default' else 'random'}_hyp_{robot_type}",
        f"% hp_tag={hp_tag}  |  robot_type={robot_type}", "",
    ]
    for alg in ALGORITHMS:
        cells = []
        for N in robot_counts:
            row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if row.empty:
                cells.append("---")
            else:
                r      = row.iloc[0]
                dagger = r"^\dagger" if r["is_best"] else ""
                s = (f"${r['mean_reward']:.1f} \\pm {r['std_reward']:.1f}"
                     f"\\ ({r['iqm']:.1f}){dagger}$")
                if r["is_best"]:
                    s = "$" + r"\mathbf{" + s[1:-1] + "}" + "$"
                cells.append(s)
        lines.append(f"{alg} & " + " & ".join(cells) + r" \\")

    out_path = os.path.join(results_dir,
                            f"main_{hp_tag}_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


def write_latex_main_combined(default_summary: pd.DataFrame,
                              tuned_summary:   pd.DataFrame,
                              results_dir:     str,
                              robot_type:      str):
    robot_counts = _robot_counts(robot_type)

    def _remark_iqm(summary):
        if summary is None: return None
        return mark_best(summary.copy(), value_col="iqm", group_cols=["num_robots"])

    def _iqm_cell(summary, alg, N):
        if summary is None: return "---"
        row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
        if row.empty: return "---"
        r   = row.iloc[0]
        val = f"{r['iqm']:.1f}"
        return (rf"$\mathbf{{{val}}}^\dagger$" if r["is_best"] else f"${val}$")

    default_iqm = _remark_iqm(default_summary)
    tuned_iqm   = _remark_iqm(tuned_summary)

    lines = [
        rf"% LaTeX table rows for tab:main_results — {robot_type} half",
        r"% IQM only, default + tuned side-by-side.",
        r"% Paste between \midrule and \bottomrule of the matching minipage.", "",
    ]
    for alg in ALGORITHMS:
        default_cells = [_iqm_cell(default_iqm, alg, N) for N in robot_counts]
        tuned_cells   = [_iqm_cell(tuned_iqm,   alg, N) for N in robot_counts]
        lines.append(f"    {alg:<6} & " + " & ".join(default_cells + tuned_cells) + r" \\")

    out_path = os.path.join(results_dir, f"main_combined_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Combined both-robot main table — tab:main_results
# Writes a single .txt with UAV rows (left minipage) then Wheeled
# rows (right minipage) so both halves can be pasted into one table.
# ================================================================

def write_latex_main_combined_both_robots(
        uav_default: pd.DataFrame, uav_tuned: pd.DataFrame,
        whl_default: pd.DataFrame, whl_tuned: pd.DataFrame,
        results_dir: str):
    """Write a single LaTeX snippet covering both halves of tab:main_results."""

    def _remark_iqm(summary):
        if summary is None:
            return None
        return mark_best(summary.copy(), value_col="iqm", group_cols=["num_robots"])

    def _iqm_cell(summary, alg, N):
        if summary is None:
            return "---"
        row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
        if row.empty:
            return "---"
        r   = row.iloc[0]
        val = f"{r['iqm']:.1f}"
        return (rf"$\mathbf{{{val}}}^\dagger$" if r["is_best"] else f"${val}$")

    uav_def_iqm = _remark_iqm(uav_default)
    uav_tun_iqm = _remark_iqm(uav_tuned)
    whl_def_iqm = _remark_iqm(whl_default)
    whl_tun_iqm = _remark_iqm(whl_tuned)

    uav_counts = _robot_counts("uav")
    whl_counts = _robot_counts("wheeled")

    lines = [
        r"% tab:main_results — combined UAV (a) + Wheeled (b) side-by-side",
        r"% Paste each block between \midrule and \bottomrule of the",
        r"% corresponding minipage in 0_main.tex.", "",
        r"% ── (a) UAV minipage rows ──────────────────────────────────────", "",
    ]
    for alg in ALGORITHMS:
        def_cells = [_iqm_cell(uav_def_iqm, alg, N) for N in uav_counts]
        tun_cells = [_iqm_cell(uav_tun_iqm, alg, N) for N in uav_counts]
        lines.append(f"    {alg:<6} & " + " & ".join(def_cells + tun_cells) + r" \\")

    lines += [
        "",
        r"% ── (b) Wheeled minipage rows ──────────────────────────────────", "",
    ]
    for alg in ALGORITHMS:
        def_cells = [_iqm_cell(whl_def_iqm, alg, N) for N in whl_counts]
        tun_cells = [_iqm_cell(whl_tun_iqm, alg, N) for N in whl_counts]
        lines.append(f"    {alg:<6} & " + " & ".join(def_cells + tun_cells) + r" \\")

    out_path = os.path.join(results_dir, "main_combined_latex_rows_both.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Reward ablation  (Step 4)
# ================================================================

def process_ablation_reward(log_root: str, results_dir: str,
                            robot_type: str = "uav") -> pd.DataFrame:
    condition_labels = {
        "full":    "Full reward",
        "no_term": r"No $R_\text{term}$",
        "no_path": r"No $R_\text{path}$",
    }
    condition_removed = {
        "full":    "---",
        "no_term": "collision penalty + success bonus",
        "no_path": "energy + speed + path + time penalty",
    }

    csv_path = os.path.join(results_dir, f"ablation_reward_{robot_type}.csv")
    has_csv  = os.path.exists(csv_path)
    df_csv   = pd.read_csv(csv_path) if has_csv else None

    env_sets = _env_sets(robot_type)

    summary_rows = []
    for cond, label in condition_labels.items():
        if has_csv and "success_pct" in df_csv.columns:
            grp = df_csv[df_csv["ablation"] == cond]
            if grp.empty: continue
            r       = grp["mean_reward"].values
            ep_lens = grp["mean_ep_length"].values

            # Dominant terminal condition
            vis = grp["success_pct"].mean()
            col = grp["collision_pct"].mean()
            mx  = grp["max_steps_pct"].mean()
            rates = {"Visited All": vis, "Collision": col, "Max Steps": mx}
            dom   = max(rates, key=rates.get)
            term_cond_str = f"{dom} ({rates[dom]:.0f}\\%)"

        else:
            version = f"ablation_reward_{cond}"
            print(f"  Scanning {log_root}/{version}/ (NPZ fallback, {robot_type})")
            rows = collect_rewards(log_root, version, ["CrossQ"], [3],
                                   env_sets, SEEDS, robot_type=robot_type)
            if not rows: continue
            r         = np.array([row["mean_reward"]    for row in rows])
            ep_lens   = np.array([row["mean_ep_length"] for row in rows])
            term_cond_str = "---"

        summary_rows.append(dict(
            condition=label,
            removed_term=condition_removed[cond],
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            iqm=compute_iqm(r),
            mean_ep_length=float(np.nanmean(ep_lens)),
            std_ep_length=float(np.nanstd(ep_lens)),
            term_cond=term_cond_str,
            raw_rewards=list(r),
        ))

    if not summary_rows:
        print(f"  [WARN] Reward ablation ({robot_type}) — no data found, skipping.")
        return None

    summary  = pd.DataFrame(summary_rows)
    out_path = os.path.join(results_dir, f"ablation_reward_agg_{robot_type}.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_reward(summary, results_dir, robot_type)
    return summary


def _write_latex_ablation_reward(summary: pd.DataFrame, results_dir: str,
                                  robot_type: str):
    lines = [f"% LaTeX table rows for tab:ablation_reward_{robot_type}", ""]
    best_ep_len = summary["mean_ep_length"].min()

    for _, r in summary.iterrows():
        bold = (r["mean_ep_length"] == best_ep_len)
        ep_s = (
            rf"\mathbf{{{r['mean_ep_length']:.1f} \pm {r['std_ep_length']:.1f}}}"
            if bold
            else rf"{r['mean_ep_length']:.1f} \pm {r['std_ep_length']:.1f}"
        )
        lines.append(
            f"{r['condition']} & {r['removed_term']} & "
            f"${r['mean_reward']:.1f}$ & ${r['std_reward']:.1f}$ & ${ep_s}$ & {r['term_cond']} \\\\"
        )
    out_path = os.path.join(results_dir,
                            f"ablation_reward_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Observation ablation  (Step 5)
# ================================================================

def process_ablation_obs(log_root: str, results_dir: str,
                         robot_type: str = "uav") -> pd.DataFrame:
    if robot_type == "uav":
        # MultiUAV: full=4N+1, no_pos=1, no_vis_hist=4N, pos_only=2N
        obs_dims = {"full": "4N+1", "no_pos": "1",
                    "no_vis_hist": "4N", "pos_only": "2N"}
    else:
        # MultiWheeled: full=5N+1, no_pos=3N+1, no_vis_hist=5N, pos_only=2N
        obs_dims = {"full": "5N+1", "no_pos": "3N+1",
                    "no_vis_hist": "5N", "pos_only": "2N"}
    labels   = {"full": "Full obs.", "no_pos": "No positions",
                "no_vis_hist": "No coverage history", "pos_only": "Positions only"}

    env_sets = _env_sets(robot_type)

    summary_rows = []
    for cond in ["full", "no_pos", "no_vis_hist", "pos_only"]:
        version = f"ablation_obs_{cond}"
        print(f"  Scanning {log_root}/{version}/  (robot_type={robot_type})")
        rows = collect_rewards(log_root, version, ["CrossQ"], [3],
                               env_sets, SEEDS, robot_type=robot_type)
        if not rows: continue
        r = np.array([row["mean_reward"] for row in rows])
        summary_rows.append(dict(
            condition=labels.get(cond, cond), obs_dim=obs_dims.get(cond, "?"),
            mean_reward=float(np.mean(r)), iqm=compute_iqm(r), raw_rewards=list(r),
        ))

    if not summary_rows: return None

    summary  = pd.DataFrame(summary_rows)
    full_row = summary[summary["condition"] == labels["full"]]
    ref_iqm  = (float(full_row["iqm"].iloc[0]) if not full_row.empty
                else (float(summary["iqm"].iloc[0]) if not summary.empty else 1.0))
    summary["delta_iqm_pct"] = ((summary["iqm"] - ref_iqm) / (abs(ref_iqm) + 1e-9) * 100)
    out_path = os.path.join(results_dir, f"ablation_obs_agg_{robot_type}.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_obs(summary, results_dir, robot_type)
    return summary


def _write_latex_ablation_obs(summary: pd.DataFrame, results_dir: str,
                               robot_type: str):
    lines = [f"% LaTeX table rows for tab:ablation_obs_{robot_type}", ""]
    for _, r in summary.iterrows():
        is_baseline = (r["condition"] == "Full obs.")
        delta  = "---" if is_baseline else f"${r['delta_iqm_pct']:.1f}$"
        mean_s = (rf"\mathbf{{{r['mean_reward']:.1f}}}" if is_baseline else f"{r['mean_reward']:.1f}")
        iqm_s  = (rf"\mathbf{{{r['iqm']:.1f}}}" if is_baseline else f"{r['iqm']:.1f}")
        lines.append(f"{r['condition']} & ${r['obs_dim']}$ & ${mean_s}$ & ${iqm_s}$ & {delta} \\\\")
    out_path = os.path.join(results_dir, f"ablation_obs_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Uncertainty ablation  (Step 6)
# ================================================================

def process_ablation_uncertainty(results_dir: str,
                                 robot_type: str = "uav") -> pd.DataFrame:
    csv_path = os.path.join(results_dir, f"ablation_uncertainty_{robot_type}.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping uncertainty ablation.")
        return None

    df = pd.read_csv(csv_path)
    df["mean_reward"] = df["mean_reward"].astype(float)
    train_modes = ["full", "wind_only", "act_only", "deterministic"]

    rows = []
    for train_mode in train_modes:
        grp_same = df[(df["ablation"] == train_mode) & (df["eval_uncertainty_mode"] == train_mode)]
        grp_full = df[(df["ablation"] == train_mode) & (df["eval_uncertainty_mode"] == "full")]
        grp_det  = df[(df["ablation"] == train_mode) & (df["eval_uncertainty_mode"] == "deterministic")]
        rows.append(dict(
            train_condition=train_mode,
            eval_same_iqm=compute_iqm(grp_same["mean_reward"].values) if not grp_same.empty else float("nan"),
            eval_full_iqm=compute_iqm(grp_full["mean_reward"].values) if not grp_full.empty else float("nan"),
            eval_det_iqm =compute_iqm(grp_det["mean_reward"].values)  if not grp_det.empty  else float("nan"),
        ))

    summary  = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, f"ablation_uncertainty_agg_{robot_type}.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_uncertainty(summary, results_dir, robot_type)
    return summary


def _write_latex_ablation_uncertainty(summary: pd.DataFrame, results_dir: str,
                                       robot_type: str):
    noise_labels = {
        "full": "wind + actuation", "wind_only": "wind",
        "act_only": "actuation", "deterministic": "none",
    }
    def _fmt(v): return "---" if (isinstance(v, float) and np.isnan(v)) else f"${v:.1f}$"
    lines = [f"% LaTeX table rows for tab:ablation_uncertainty_{robot_type}", ""]
    for _, r in summary.iterrows():
        tc = r["train_condition"]
        lines.append(f"{tc.replace('_',' ').title()} & {noise_labels.get(tc,'?')} & "
                     f"{_fmt(r['eval_same_iqm'])} & {_fmt(r['eval_full_iqm'])} & {_fmt(r['eval_det_iqm'])} \\\\")
    out_path = os.path.join(results_dir,
                            f"ablation_uncertainty_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Domain randomization  (Step 7)
# ================================================================

def process_dr(results_dir: str, robot_type: str = "uav") -> pd.DataFrame:
    in_dist_path = os.path.join(results_dir, f"dr_inDist_{robot_type}.csv")
    ood_path     = os.path.join(results_dir, f"dr_OOD_{robot_type}.csv")
    if not os.path.exists(in_dist_path) or not os.path.exists(ood_path):
        print(f"  [WARN] DR CSVs not found for {robot_type} "
              f"({in_dist_path}, {ood_path}) — skipping.")
        return None

    df_in  = pd.read_csv(in_dist_path)
    df_ood = pd.read_csv(ood_path)
    df_in["mean_reward"]  = df_in["mean_reward"].astype(float)
    df_ood["mean_reward"] = df_ood["mean_reward"].astype(float)

    dr_conditions = [("none", "(A) no DR", "none (standard training)"),
                     ("wind", "(B) wind DR", "wind speed + direction"),
                     ("full", "(C) full DR", "all parameters")]

    rows = []
    for ablation, label, rand_params in dr_conditions:
        r_in  = df_in[df_in["ablation"]  == ablation]["mean_reward"].values
        r_ood = df_ood[df_ood["ablation"] == ablation]["mean_reward"].values
        all_r = np.concatenate([r_in, r_ood]) if len(r_in) and len(r_ood) else np.array([])
        rows.append(dict(
            condition=label, rand_params=rand_params,
            in_dist_iqm=compute_iqm(r_in) if len(r_in) else float("nan"),
            ood_iqm=compute_iqm(r_ood) if len(r_ood) else float("nan"),
            cvar_0_1=cvar_0_1(all_r) if len(all_r) else float("nan"),
        ))

    summary  = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, f"dr_results_agg_{robot_type}.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_dr(summary, results_dir, robot_type)
    return summary


def _write_latex_dr(summary: pd.DataFrame, results_dir: str, robot_type: str):
    lines = [f"% LaTeX table rows for tab:dr_results_{robot_type}", ""]
    def _fmt(v, bold=False):
        if isinstance(v, float) and np.isnan(v): return "---"
        s = f"{v:.1f}"
        return (rf"\mathbf{{{s}}}^\dagger" if bold else s)

    valid_ood  = summary["ood_iqm"].dropna()
    valid_cvar = summary["cvar_0_1"].dropna()
    best_ood   = valid_ood.max()  if not valid_ood.empty  else None
    best_cvar  = valid_cvar.max() if not valid_cvar.empty else None

    for _, r in summary.iterrows():
        ood_s  = _fmt(r["ood_iqm"],  bold=(best_ood  is not None and r["ood_iqm"]  == best_ood))
        cvar_s = _fmt(r["cvar_0_1"], bold=(best_cvar is not None and r["cvar_0_1"] == best_cvar))
        lines.append(f"{r['condition']} & {r['rand_params']} & "
                     f"${_fmt(r['in_dist_iqm'])}$ & ${ood_s}$ & ${cvar_s}$ \\\\")
    out_path = os.path.join(results_dir, f"dr_results_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Combined ablations + DR table — tab:ablations_and_dr
# ================================================================

def write_latex_combined_ablations_dr(reward_df: pd.DataFrame, obs_df: pd.DataFrame,
                                       unc_df: pd.DataFrame, dr_df: pd.DataFrame,
                                       results_dir: str, robot_type: str):
    OBS_LABEL_MAP = {"Full obs.": "Full", "No positions": "No pos.",
                     "No coverage history": "No cov.", "Positions only": "Pos. only"}
    UNC_LABEL_MAP = {"full": "Full", "wind_only": "Wind only",
                     "act_only": "Act. only", "deterministic": "Deterministic"}

    unc_simple = None
    if unc_df is not None:
        unc_rows = []
        for _, r in unc_df.iterrows():
            tc = r["train_condition"]
            unc_rows.append(dict(condition=UNC_LABEL_MAP.get(tc, tc),
                                 iqm=float(r.get("eval_same_iqm", float("nan")))))
        unc_simple = pd.DataFrame(unc_rows)

    def _safe_val(df, col, i):
        if df is None or i >= len(df): return float("nan")
        v = df[col].iloc[i]
        return float("nan") if (isinstance(v, float) and np.isnan(v)) else v

    def _fmt(val, is_best=False):
        if isinstance(val, float) and np.isnan(val): return ""
        s = f"{val:.1f}"
        return (rf"$\mathbf{{{s}}}^\dagger$" if is_best else f"${s}$")

    def _best_mask_iqm(df):
        if df is None or df.empty: return []
        vals = df["iqm"].values
        sorted_idx = np.argsort(vals)[::-1]
        mask = [False] * len(df)
        if len(sorted_idx) >= 2 and "raw_rewards" in df.columns:
            a, b = np.array(df["raw_rewards"].iloc[sorted_idx[0]]), np.array(df["raw_rewards"].iloc[sorted_idx[1]])
            if mannwhitney_pval(a, b) < 0.05: mask[sorted_idx[0]] = True
        elif sorted_idx.size >= 1:
            mask[sorted_idx[0]] = True
        return mask

    def _best_mask_ep_len(df):
        if df is None or df.empty or "mean_ep_length" not in df.columns: return []
        vals = df["mean_ep_length"].values
        mask = [False] * len(df)
        if not np.all(np.isnan(vals)): mask[int(np.nanargmin(vals))] = True
        return mask

    def _dr_best_mask(col):
        if dr_df is None or col not in dr_df.columns:
            return [False] * (len(dr_df) if dr_df is not None else 0)
        vals = dr_df[col].values
        m    = [False] * len(dr_df)
        if not np.all(np.isnan(vals)): m[int(np.nanargmax(vals))] = True
        return m

    rew_best  = _best_mask_ep_len(reward_df)
    obs_best  = _best_mask_iqm(obs_df)
    unc_best  = [False] * (len(unc_simple) if unc_simple is not None else 0)
    if unc_simple is not None and not unc_simple.empty:
        non_nan = ~np.isnan(unc_simple["iqm"].values)
        if non_nan.any(): unc_best[int(np.nanargmax(unc_simple["iqm"].values))] = True

    dr_ood_best  = _dr_best_mask("ood_iqm")
    dr_cvar_best = _dr_best_mask("cvar_0_1")

    lines = [rf"% LaTeX table rows for tab:ablations_and_dr_{robot_type}",
             r"% Paste between \midrule and \bottomrule.", ""]
    for i in range(4):
        r_cond   = (str(_safe_val(reward_df, "condition", i))
                    if reward_df is not None and i < len(reward_df) else "")
        r_ep_len = _fmt(_safe_val(reward_df, "mean_ep_length", i),
                        rew_best[i] if i < len(rew_best) else False)
        r_term   = (str(reward_df["term_cond"].iloc[i])
                    if reward_df is not None and "term_cond" in reward_df.columns
                    and i < len(reward_df) else "")

        raw_o  = (str(_safe_val(obs_df, "condition", i))
                  if obs_df is not None and i < len(obs_df) else "")
        o_cond = OBS_LABEL_MAP.get(raw_o, raw_o)
        o_iqm  = _fmt(_safe_val(obs_df, "iqm", i), obs_best[i] if i < len(obs_best) else False)

        u_cond = (str(_safe_val(unc_simple, "condition", i))
                  if unc_simple is not None and i < len(unc_simple) else "")
        u_iqm  = _fmt(_safe_val(unc_simple, "iqm", i), unc_best[i] if i < len(unc_best) else False)

        if i < 3 and dr_df is not None and i < len(dr_df):
            d_cond    = str(dr_df["condition"].iloc[i])
            d_indist  = _fmt(dr_df["in_dist_iqm"].iloc[i], False)
            d_ood     = _fmt(dr_df["ood_iqm"].iloc[i],
                             dr_ood_best[i] if i < len(dr_ood_best) else False)
            d_cvar    = _fmt(dr_df["cvar_0_1"].iloc[i],
                             dr_cvar_best[i] if i < len(dr_cvar_best) else False)
        else:
            d_cond, d_indist, d_ood, d_cvar = "", "", "", ""

        lines.append(
            f"    {r_cond} & {r_ep_len} & {r_term} & {o_cond} & {o_iqm} & "
            f"{u_cond} & {u_iqm} & {d_cond} & {d_indist} & {d_ood} & {d_cvar} \\\\"
        )

    out_path = os.path.join(results_dir,
                            f"combined_ablations_dr_latex_rows_{robot_type}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Console summary
# ================================================================

def print_summary_table(summary: pd.DataFrame, label: str, hp_tag: str,
                        robot_type: str = "uav"):
    if summary is None: return
    robot_counts = _robot_counts(robot_type)
    print(f"\n{'=' * 80}\n  {label} ({hp_tag}, {robot_type})\n{'=' * 80}")
    header = f"{'Algorithm':10s}" + "".join(f"  N={N:5}" for N in robot_counts)
    print(header)
    print("-" * len(header))
    for alg in ALGORITHMS:
        row_str = f"{alg:10s}"
        for N in robot_counts:
            r = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if r.empty:
                row_str += "     ---   "
            else:
                ri   = r.iloc[0]
                flag = "*" if ri["is_best"] else " "
                row_str += f"  {ri['mean_reward']:6.1f}{flag}"
        print(row_str)
    print(f"{'=' * 80}")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate logs into LaTeX-ready tables.")
    p.add_argument("--log_root",    type=str, default="logs")
    p.add_argument("--results_dir", type=str, default=r"logs/results")
    p.add_argument("--robot_type",  type=str, default="uav",
                   choices=["uav", "wheeled", "both"],
                   help="Process one platform or 'both' to also emit "
                        "main_combined_latex_rows_both.txt for tab:main_results.")
    return p.parse_args()


def _process_one_robot(log_root: str, results_dir: str, robot_type: str):
    """Run all analyses for a single robot type and return summary dicts."""
    print(f"\n{'=' * 60}")
    print(f"  Processing robot_type = {robot_type}")
    print(f"{'=' * 60}")

    print("\n-- Main results: default HPs -------------------------------------------")
    default_summary = process_main(log_root, results_dir, "default",
                                   robot_type=robot_type)
    print_summary_table(default_summary, "Main results", "default",
                        robot_type=robot_type)

    print("\n-- Main results: tuned HPs ---------------------------------------------")
    tuned_summary = process_main(log_root, results_dir, "tuned",
                                 robot_type=robot_type)
    print_summary_table(tuned_summary, "Main results", "tuned",
                        robot_type=robot_type)

    print("\n-- Reward ablation -----------------------------------------------------")
    reward_summary = process_ablation_reward(log_root, results_dir,
                                             robot_type=robot_type)

    print("\n-- Observation ablation ------------------------------------------------")
    obs_summary = process_ablation_obs(log_root, results_dir,
                                       robot_type=robot_type)

    print("\n-- Uncertainty ablation ------------------------------------------------")
    unc_summary = process_ablation_uncertainty(results_dir, robot_type=robot_type)

    print("\n-- Domain randomization ------------------------------------------------")
    dr_summary = process_dr(results_dir, robot_type=robot_type)

    write_latex_main_combined(default_summary, tuned_summary,
                              results_dir, robot_type=robot_type)
    write_latex_combined_ablations_dr(reward_summary, obs_summary,
                                      unc_summary, dr_summary,
                                      results_dir, robot_type=robot_type)
    return default_summary, tuned_summary


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nLog root    : {args.log_root}")
    print(f"Results dir : {args.results_dir}")
    print(f"Robot type  : {args.robot_type}")

    if args.robot_type == "both":
        uav_def, uav_tun = _process_one_robot(args.log_root, args.results_dir, "uav")
        whl_def, whl_tun = _process_one_robot(args.log_root, args.results_dir, "wheeled")

        print("\n-- Merged side-by-side table (tab:main_results) --------------------")
        write_latex_main_combined_both_robots(
            uav_def, uav_tun, whl_def, whl_tun, args.results_dir)
    else:
        _process_one_robot(args.log_root, args.results_dir, args.robot_type)

    print("\n✓ analyze_results.py complete.")


if __name__ == "__main__":
    main()
