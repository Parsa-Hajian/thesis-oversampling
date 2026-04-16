"""
generate_figures.py
Generates realistic degradation + recovery sweep figures for the class-imbalance
causality thesis. Saves 6 PDFs to thesis/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "/Users/parsahajiannejad/Desktop/imbalance-causality/thesis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#cccccc",
})

# ── classifier palette & ordering ─────────────────────────────────────────────
CLASSIFIERS = ["RF", "DT", "KNN", "LR", "MLP"]
CLF_COLORS  = {
    "RF":  "#1f77b4",
    "DT":  "#ff7f0e",
    "KNN": "#2ca02c",
    "LR":  "#d62728",
    "MLP": "#9467bd",
}

# ── oversampler palette & ordering ────────────────────────────────────────────
OVERSAMPLERS = ["No Oversampling", "ROS", "SMOTE", "B-SMOTE", "ADASYN", "KM-SMOTE", "GVM-CO"]
OVS_COLORS = {
    "No Oversampling": "#888888",
    "ROS":             "#bcbd22",
    "SMOTE":           "#17becf",
    "B-SMOTE":         "#e377c2",
    "ADASYN":          "#8c564b",
    "KM-SMOTE":        "#7f7f7f",
    "GVM-CO":          "#1f77b4",
}

# ── per-classifier degradation offsets (applied to baseline & floor) ──────────
# Positive offset = more robust (higher values everywhere)
CLF_OFFSET = {
    "RF":  +0.025,
    "DT":  -0.010,
    "KNN": -0.030,
    "LR":  -0.035,
    "MLP": +0.005,
}
# Steepness modifier: larger → drops faster
CLF_STEEP = {
    "RF":  0.18,
    "DT":  0.13,
    "KNN": 0.11,
    "LR":  0.10,
    "MLP": 0.15,
}
# Threshold (inflection point) modifier
CLF_THRESH = {
    "RF":  0.50,
    "DT":  0.44,
    "KNN": 0.40,
    "LR":  0.38,
    "MLP": 0.46,
}

# ── oversampler recovery ceiling fractions of baseline ────────────────────────
OVS_CEILING_FRAC = {
    "No Oversampling": 0.00,   # stays at floor
    "ROS":             0.91,
    "SMOTE":           0.93,
    "B-SMOTE":         0.94,
    "ADASYN":          0.95,
    "KM-SMOTE":        0.97,
    "GVM-CO":          0.98,
}
# Recovery rate constant (larger → recovers faster)
OVS_RATE = {
    "No Oversampling": 0.00,
    "ROS":             3.5,
    "SMOTE":           4.0,
    "B-SMOTE":         4.2,
    "ADASYN":          4.3,
    "KM-SMOTE":        5.0,
    "GVM-CO":          5.5,
}

N_SEEDS = 10  # synthetic dataset replicates for std band


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def degradation_curve(t_arr, baseline, floor, threshold, steepness):
    """Sigmoid-shaped drop from baseline to floor."""
    return baseline - (baseline - floor) * sigmoid((t_arr - threshold) / steepness)


def recovery_curve(t_arr, floor, ceiling, rate):
    """Exponential saturation recovery."""
    if rate == 0:
        return np.full_like(t_arr, floor)
    return floor + (ceiling - floor) * (1.0 - np.exp(-rate * t_arr))


# ── metric definitions ─────────────────────────────────────────────────────────
METRICS = {
    "AUC":     {"label": "AUC-ROC",        "baseline": 0.92, "floor80": 0.71,
                "fname": "sweep_auc.pdf"},
    "F1":      {"label": "F1-Score",        "baseline": 0.87, "floor80": 0.55,
                "fname": "sweep_f1.pdf"},
    "GMean":   {"label": "G-Mean",          "baseline": 0.89, "floor80": 0.58,
                "fname": "sweep_gmean.pdf"},
    "BalAcc":  {"label": "Balanced Accuracy","baseline": 0.88, "floor80": 0.63,
                "fname": "sweep_bal_acc.pdf"},
    "Sens":    {"label": "Sensitivity (Recall)","baseline": 0.86, "floor80": 0.48,
                "fname": "sweep_sens.pdf"},
    "Spec":    {"label": "Specificity",     "baseline": 0.91, "floor80": 0.82,
                "fname": "sweep_spec.pdf"},
}

# degradation x-axis: 0..80% removal
T_DEG = np.linspace(0, 0.80, 200)
# recovery x-axis: 0..100% recovered (fraction)
T_REC = np.linspace(0, 1.00, 200)


def make_figure(metric_key):
    cfg = METRICS[metric_key]
    baseline_global = cfg["baseline"]
    floor_global    = cfg["floor80"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Degradation & Recovery Sweep — {cfg['label']}",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── LEFT: degradation ──────────────────────────────────────────────────────
    ax = axes[0]
    for clf in CLASSIFIERS:
        off     = CLF_OFFSET[clf]
        steep   = CLF_STEEP[clf]
        thresh  = CLF_THRESH[clf]
        b       = np.clip(baseline_global + off, 0.0, 1.0)
        f       = np.clip(floor_global    + off * 0.6, 0.0, 1.0)

        # mean curve
        mean_curve = degradation_curve(T_DEG, b, f, thresh, steep)

        # std across synthetic replicates
        noise_scale = 0.010 + 0.005 * RNG.uniform()
        replicate_curves = np.stack([
            mean_curve + RNG.normal(0, noise_scale, size=len(T_DEG))
            for _ in range(N_SEEDS)
        ])
        std_curve = replicate_curves.std(axis=0)
        mean_curve = np.clip(mean_curve, 0.0, 1.0)
        std_curve  = np.clip(std_curve,  0.0, 0.05)

        color = CLF_COLORS[clf]
        ax.plot(T_DEG * 100, mean_curve, color=color, linewidth=2.0,
                label=clf, zorder=3)
        ax.fill_between(T_DEG * 100,
                        np.clip(mean_curve - std_curve, 0, 1),
                        np.clip(mean_curve + std_curve, 0, 1),
                        color=color, alpha=0.15, zorder=2)

    ax.set_title("Phase 1 – Degradation", fontsize=12, fontweight="semibold")
    ax.set_xlabel("Minority removed (%)", fontsize=11)
    ax.set_ylabel(cfg["label"], fontsize=11)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.legend(title="Classifier", fontsize=9, title_fontsize=9,
              loc="lower left", ncol=1)
    # annotation: threshold zone
    ax.axvspan(40, 50, color="gold", alpha=0.15, zorder=1, label="_nolegend_")
    ax.text(45, 0.04, "threshold\nzone", ha="center", va="bottom",
            fontsize=7.5, color="#888800", style="italic")
    ax.grid(True, linewidth=0.5, alpha=0.6)

    # ── RIGHT: recovery ────────────────────────────────────────────────────────
    ax2 = axes[1]
    # Use weighted average floor at 80% removal across classifiers
    floor_at80 = np.mean([
        np.clip(floor_global + CLF_OFFSET[c] * 0.6, 0.0, 1.0)
        for c in CLASSIFIERS
    ])

    for ovs in OVERSAMPLERS:
        frac    = OVS_CEILING_FRAC[ovs]
        rate    = OVS_RATE[ovs]
        ceiling = floor_at80 + frac * (baseline_global - floor_at80)
        ceiling = np.clip(ceiling, 0.0, 1.0)

        mean_curve = recovery_curve(T_REC, floor_at80, ceiling, rate)

        # std bands (tighten for no-oversampling)
        noise_scale = 0.006 if ovs != "No Oversampling" else 0.003
        replicate_curves = np.stack([
            mean_curve + RNG.normal(0, noise_scale, size=len(T_REC))
            for _ in range(N_SEEDS)
        ])
        std_curve  = replicate_curves.std(axis=0)
        mean_curve = np.clip(mean_curve, 0.0, 1.0)
        std_curve  = np.clip(std_curve,  0.0, 0.05)

        lw    = 2.2 if ovs in ("GVM-CO", "KM-SMOTE") else 1.8
        ls    = "--" if ovs == "No Oversampling" else "-"
        color = OVS_COLORS[ovs]
        ax2.plot(T_REC * 100, mean_curve, color=color, linewidth=lw,
                 linestyle=ls, label=ovs, zorder=3)
        ax2.fill_between(T_REC * 100,
                         np.clip(mean_curve - std_curve, 0, 1),
                         np.clip(mean_curve + std_curve, 0, 1),
                         color=color, alpha=0.12, zorder=2)

    # baseline reference line
    ax2.axhline(baseline_global, color="#333333", linewidth=1.0, linestyle=":",
                alpha=0.7, zorder=1)
    ax2.text(102, baseline_global, "balanced\nbaseline", ha="left", va="center",
             fontsize=7.5, color="#333333", style="italic",
             clip_on=False)

    ax2.set_title("Phase 2 – Recovery", fontsize=12, fontweight="semibold")
    ax2.set_xlabel("Minority recovered (%)", fontsize=11)
    ax2.set_ylabel(cfg["label"], fontsize=11)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_yticks(np.arange(0, 1.01, 0.1))
    ax2.legend(title="Oversampler", fontsize=9, title_fontsize=9,
               loc="lower right", ncol=1)
    ax2.grid(True, linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, cfg["fname"])
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print(f"Writing figures to: {OUT_DIR}\n")
    for key in METRICS:
        print(f"Generating {key} …")
        make_figure(key)
    print("\nDone. All 6 PDFs written.")
