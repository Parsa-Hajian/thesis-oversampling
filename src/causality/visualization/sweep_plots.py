"""
sweep_plots.py -- Signature V/U-shape degradation + recovery figures.

One figure per metric (6 total).  Each figure has two panels:
  - Left panel  (degradation): x-axis 0% → 80% removed, y = metric
  - Right panel (recovery):    x-axis 80% → 0% removed, y = metric

Usage:
    cd ~/Desktop/imbalance-causality
    /opt/anaconda3/bin/python3 src/visualization/sweep_plots.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

METRICS = {
    "auc":     "AUC-ROC",
    "f1":      "F1-Score",
    "gmean":   "G-Mean",
    "bal_acc": "Balanced Accuracy",
    "sens":    "Sensitivity",
    "spec":    "Specificity",
}

CLF_COLORS = {
    "RF":  "#1f77b4",
    "DT":  "#ff7f0e",
    "KNN": "#2ca02c",
    "LR":  "#d62728",
    "MLP": "#9467bd",
}

OVS_COLORS = {
    "none":   "#888888",
    "ros":    "#bcbd22",
    "smote":  "#17becf",
    "bsmote": "#e377c2",
    "adasyn": "#8c564b",
    "kmsmote":"#7f7f7f",
    "gvmco":  "#1f77b4",
    "lreco":  "#ff7f0e",
    "lsco":   "#2ca02c",
}

OVS_LABELS = {
    "none":   "No Oversampling",
    "ros":    "ROS",
    "smote":  "SMOTE",
    "bsmote": "B-SMOTE",
    "adasyn": "ADASYN",
    "kmsmote":"KM-SMOTE",
    "gvmco":  "GVM-CO",
    "lreco":  "LRE-CO",
    "lsco":   "LS-CO",
}


def _load_degradation():
    deg_dir = os.path.join(ROOT, "results", "degradation")
    dfs = []
    if not os.path.isdir(deg_dir):
        return pd.DataFrame()
    for f in os.listdir(deg_dir):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(deg_dir, f)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_recovery():
    rec_dir = os.path.join(ROOT, "results", "recovery")
    dfs = []
    if not os.path.isdir(rec_dir):
        return pd.DataFrame()
    for f in os.listdir(rec_dir):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(rec_dir, f)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _mean_std_curve(df, x_col, group_col, metric):
    """Return dict: group -> (x_vals, mean_vals, std_vals)."""
    result = {}
    for grp, gdf in df.groupby(group_col):
        agg = (gdf.groupby(x_col)[metric]
               .agg(["mean", "std"])
               .reset_index()
               .sort_values(x_col))
        result[grp] = (
            agg[x_col].values,
            agg["mean"].values,
            agg["std"].fillna(0).values,
        )
    return result


def plot_signature_figure(metric_key: str, metric_label: str,
                          deg_df: pd.DataFrame, rec_df: pd.DataFrame,
                          out_path: str):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.35)
    ax_deg = fig.add_subplot(gs[0])
    ax_rec = fig.add_subplot(gs[1])

    # ---- Degradation panel ----
    if not deg_df.empty and metric_key in deg_df.columns:
        curves = _mean_std_curve(deg_df, "removal_frac", "classifier", metric_key)
        for clf, (x, m, s) in curves.items():
            color = CLF_COLORS.get(clf, "black")
            ax_deg.plot(x * 100, m, label=clf, color=color, linewidth=2)
            ax_deg.fill_between(x * 100, m - s, m + s,
                                alpha=0.15, color=color)
    ax_deg.set_xlabel("Minority removed (%)", fontsize=11)
    ax_deg.set_ylabel(metric_label, fontsize=11)
    ax_deg.set_title("Phase 1 – Degradation", fontsize=12, fontweight="bold")
    ax_deg.set_xlim(0, 80)
    ax_deg.set_ylim(0, 1.05)
    ax_deg.legend(fontsize=9, loc="lower left")
    ax_deg.grid(alpha=0.3)

    # ---- Recovery panel ----
    if not rec_df.empty and metric_key in rec_df.columns:
        # Average over classifiers first, then plot per oversampler
        rec_avg = (rec_df.groupby(["oversampler", "recovery_frac"])[metric_key]
                   .mean().reset_index())
        curves = _mean_std_curve(rec_df, "recovery_frac", "oversampler", metric_key)
        for ovs, (x, m, s) in curves.items():
            color = OVS_COLORS.get(ovs, "black")
            label = OVS_LABELS.get(ovs, ovs)
            ax_rec.plot(x * 100, m, label=label, color=color, linewidth=2)
            ax_rec.fill_between(x * 100, m - s, m + s,
                                alpha=0.15, color=color)
    ax_rec.set_xlabel("Minority recovered (%)", fontsize=11)
    ax_rec.set_ylabel(metric_label, fontsize=11)
    ax_rec.set_title("Phase 2 – Recovery", fontsize=12, fontweight="bold")
    ax_rec.set_xlim(0, 100)
    ax_rec.set_ylim(0, 1.05)
    ax_rec.legend(fontsize=8, loc="lower right", ncol=2)
    ax_rec.grid(alpha=0.3)

    fig.suptitle(f"Imbalance Effect on {metric_label}", fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run():
    out_dir = os.path.join(ROOT, "thesis", "figures")
    os.makedirs(out_dir, exist_ok=True)

    deg_df = _load_degradation()
    rec_df = _load_recovery()

    if deg_df.empty:
        print("No degradation results found — run run_degradation.py first.")
    if rec_df.empty:
        print("No recovery results found — run run_recovery.py first.")

    for metric_key, metric_label in METRICS.items():
        out_path = os.path.join(out_dir, f"sweep_{metric_key}.pdf")
        plot_signature_figure(metric_key, metric_label, deg_df, rec_df, out_path)


if __name__ == "__main__":
    print("Generating signature sweep figures...")
    run()
    print("Done.")
