#!/usr/bin/env python3
"""
Generate step-by-step PDF visualizations for LS-CO algorithm (generic + cluster-based).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import os

SEED = 42
DPI = 300
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 8.5, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
    'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

C_MAJ = '#4878CF'; C_MIN = '#D65F5F'; C_SYN = '#2CA02C'
C_BORDER = '#FF8C00'; C_DARK = '#333333'
C_PURPLE = '#9467BD'; C_TEAL = '#17BECF'
LAYER_COLORS = ['#FFF7BC', '#FEE391', '#FEC44F', '#FE9929', '#D95F0E']
CLUSTER_COLORS = ['#D65F5F', '#E6A817', '#9467BD']


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  Saved {path}')


def make_minority(seed=SEED):
    rng = np.random.RandomState(seed)
    min1 = rng.randn(10, 2) * 0.4 + np.array([0.5, 1.0])
    min2 = rng.randn(7, 2) * 0.35 + np.array([-0.3, 0.3])
    min3 = rng.randn(5, 2) * 0.25 + np.array([1.0, -0.1])
    return np.vstack([min1, min2, min3])


def make_dataset(seed=SEED):
    rng = np.random.RandomState(seed)
    maj1 = rng.randn(25, 2) * 0.7 + np.array([3.0, 3.0])
    maj2 = rng.randn(20, 2) * 0.6 + np.array([-1.5, -0.5])
    X_maj = np.vstack([maj1, maj2])
    X_min = make_minority(seed)
    return X_maj, X_min


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Center and enclosing circle
# ═══════════════════════════════════════════════════════════════════════════
def ls_step1():
    X_maj, X_min = make_dataset()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.15, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=55, alpha=0.9, edgecolors='black',
               linewidth=0.4, marker='s', zorder=3)
    ax.scatter(*center, c='black', marker='+', s=150, linewidths=2.5, zorder=5)

    # Enclosing circle
    circ = plt.Circle(center, R, fill=False, edgecolor=C_DARK, linewidth=1.5,
                      linestyle='-', zorder=2)
    ax.add_patch(circ)

    # Radius line
    farthest = X_min[np.argmax(np.linalg.norm(X_min - center, axis=1))]
    ax.plot([center[0], farthest[0]], [center[1], farthest[1]], c=C_BORDER,
            linewidth=1.5, linestyle=':', zorder=4)
    mid = (center + farthest) / 2
    ax.text(mid[0] + 0.1, mid[1] + 0.1, f'$R = {R:.2f}$', fontsize=9,
            color=C_BORDER, fontweight='bold')

    ax.annotate('Center $\\bar{x}$', xy=center, xytext=(center[0]+0.5, center[1]+0.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Step 1: Minority Center & Enclosing Circle', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    savefig(fig, 'ls_step1_center.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Concentric ring layers
# ═══════════════════════════════════════════════════════════════════════════
def ls_step2():
    X_min = make_minority()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))
    n_layers = 5
    boundaries = np.linspace(0, R, n_layers + 1)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Draw layers as colored rings
    theta = np.linspace(0, 2*np.pi, 200)
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.4, edgecolor=C_DARK,
                      linewidth=0.6, zorder=1)
        ax.add_patch(wedge)
        # Layer label
        r_mid = (r_in + r_out) / 2
        ax.text(center[0] + r_mid, center[1] - 0.08, f'$L_{l+1}$',
                fontsize=8, ha='center', fontweight='bold', color=C_DARK)

    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=55, alpha=0.9, edgecolors='black',
               linewidth=0.5, marker='s', zorder=3)
    ax.scatter(*center, c='black', marker='+', s=150, linewidths=2.5, zorder=5)

    # Boundary labels
    for l in range(n_layers + 1):
        r = boundaries[l]
        ax.text(center[0] + r + 0.05, center[1] + R * 0.5, f'$r_{l}$={r:.2f}',
                fontsize=6.5, color=C_DARK, rotation=0)

    ax.set_title(f'Step 2: Concentric Ring Layers ($L = {n_layers}$)', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    ax.set_xlim(center[0]-R-0.5, center[0]+R+1.2)
    ax.set_ylim(center[1]-R-0.5, center[1]+R+0.8)
    savefig(fig, 'ls_step2_layers.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Angular segments
# ═══════════════════════════════════════════════════════════════════════════
def ls_step3():
    X_min = make_minority()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))
    n_layers = 5
    boundaries = np.linspace(0, R, n_layers + 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Draw light layers
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.15, edgecolor=C_DARK,
                      linewidth=0.3, zorder=1)
        ax.add_patch(wedge)

    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, alpha=0.7, edgecolors='black',
               linewidth=0.3, marker='s', zorder=3)

    # Draw radial directions from center through each minority point
    for i in range(len(X_min)):
        direction = X_min[i] - center
        angle = np.arctan2(direction[1], direction[0])
        r_pt = np.linalg.norm(direction)
        # Find layer
        layer_idx = np.searchsorted(boundaries[1:], r_pt)
        layer_idx = min(layer_idx, n_layers - 1)
        r_in, r_out = boundaries[layer_idx], boundaries[layer_idx + 1]

        # Draw segment as a thin wedge
        ax.plot([center[0], center[0] + R * 1.05 * np.cos(angle)],
                [center[1], center[1] + R * 1.05 * np.sin(angle)],
                c=C_DARK, linewidth=0.3, alpha=0.3, linestyle=':', zorder=2)

    # Highlight one segment in detail
    highlight_idx = 3
    direction = X_min[highlight_idx] - center
    angle = np.arctan2(direction[1], direction[0])
    r_pt = np.linalg.norm(direction)
    layer_idx = np.searchsorted(boundaries[1:], r_pt)
    layer_idx = min(layer_idx, n_layers - 1)
    r_in, r_out = boundaries[layer_idx], boundaries[layer_idx + 1]

    # Highlighted segment wedge
    ang_width = 12  # degrees
    ang_deg = np.degrees(angle)
    wedge = Wedge(center, r_out, ang_deg - ang_width, ang_deg + ang_width,
                  width=r_out - r_in, facecolor=C_SYN, alpha=0.3, edgecolor=C_SYN,
                  linewidth=1.5, zorder=2)
    ax.add_patch(wedge)

    # Arrow showing segment direction
    ax.annotate('', xy=center + (r_out + 0.15) * np.array([np.cos(angle), np.sin(angle)]),
                xytext=center,
                arrowprops=dict(arrowstyle='->', color=C_SYN, lw=2.0))
    ax.text(center[0] + (r_out + 0.3) * np.cos(angle),
            center[1] + (r_out + 0.3) * np.sin(angle),
            'Segment $S_i$', fontsize=8, color=C_SYN, fontweight='bold', ha='center')

    # r_in, r_out annotation
    ax.annotate(f'$r_{{in}}={r_in:.2f}$', xy=center + r_in * np.array([np.cos(angle), np.sin(angle)]),
                xytext=(center[0] - 0.8, center[1] - 0.8), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=C_DARK),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow', alpha=0.8))
    ax.annotate(f'$r_{{out}}={r_out:.2f}$', xy=center + r_out * np.array([np.cos(angle), np.sin(angle)]),
                xytext=(center[0] + 1.0, center[1] - 0.8), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=C_DARK),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow', alpha=0.8))

    ax.scatter(*center, c='black', marker='+', s=150, linewidths=2.5, zorder=5)
    ax.set_title('Step 3: Angular Segments (one highlighted)', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    savefig(fig, 'ls_step3_segments.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Per-layer allocation
# ═══════════════════════════════════════════════════════════════════════════
def ls_step4():
    X_min = make_minority()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))
    n_layers = 5
    boundaries = np.linspace(0, R, n_layers + 1)

    # Count points per layer and compute area
    dists = np.linalg.norm(X_min - center, axis=1)
    layer_counts = []
    layer_areas = []
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        count = np.sum((dists >= r_in) & (dists < r_out))
        area = np.pi * (r_out**2 - r_in**2)
        layer_counts.append(count)
        layer_areas.append(area)

    # Allocation proportional to area (area-corrected)
    total_synth = 30
    area_arr = np.array(layer_areas)
    alloc = (area_arr / area_arr.sum() * total_synth).astype(int)
    alloc[-1] = total_synth - alloc[:-1].sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: layer visualization with counts
    ax1 = axes[0]
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.4, edgecolor=C_DARK,
                      linewidth=0.6, zorder=1)
        ax1.add_patch(wedge)
        r_mid = (r_in + r_out) / 2
        ax1.text(center[0] + r_mid * 0.7, center[1] + r_mid * 0.7,
                 f'$n={layer_counts[l]}$', fontsize=7.5, ha='center',
                 fontweight='bold', color=C_DARK,
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    ax1.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=50, alpha=0.8,
                edgecolors='black', linewidth=0.4, marker='s', zorder=3)
    ax1.scatter(*center, c='black', marker='+', s=120, linewidths=2, zorder=5)
    ax1.set_title('(a) Points per Layer', fontweight='bold', fontsize=10)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.1)
    ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')

    # Right: bar chart of allocation
    ax2 = axes[1]
    x_pos = np.arange(n_layers)
    bars1 = ax2.bar(x_pos - 0.2, layer_counts, 0.35, color=C_MIN, alpha=0.7,
                    edgecolor='black', linewidth=0.5, label='Points in layer')
    bars2 = ax2.bar(x_pos + 0.2, alloc, 0.35, color=C_SYN, alpha=0.7,
                    edgecolor='black', linewidth=0.5, label='Allocated synthetics')
    for i, (c, a) in enumerate(zip(layer_counts, alloc)):
        ax2.text(i - 0.2, c + 0.3, str(c), ha='center', fontsize=8, fontweight='bold')
        ax2.text(i + 0.2, a + 0.3, str(a), ha='center', fontsize=8, fontweight='bold',
                 color=C_SYN)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'$L_{l+1}$' for l in range(n_layers)])
    ax2.set_ylabel('Count')
    ax2.set_title('(b) Area-Corrected Allocation', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.15, axis='y')

    # Add area annotation
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_pos, layer_areas, 'o--', color=C_BORDER, linewidth=1.5,
                  markersize=6, label='Layer area')
    ax2_twin.set_ylabel('Area', color=C_BORDER)
    ax2_twin.tick_params(axis='y', labelcolor=C_BORDER)
    ax2_twin.legend(loc='upper right', fontsize=7.5)

    fig.suptitle('Step 4: Per-Layer Allocation (Area-Corrected)', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'ls_step4_allocation.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Angular noise
# ═══════════════════════════════════════════════════════════════════════════
def ls_step5():
    rng = np.random.RandomState(SEED)
    center = np.array([0.0, 0.0])
    original_angle = np.radians(45)
    ang_std = np.radians(15)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Reference circle
    R = 2.0
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), c=C_DARK, linewidth=0.5, alpha=0.2)

    # Original direction
    ax.annotate('', xy=R * 1.1 * np.array([np.cos(original_angle), np.sin(original_angle)]),
                xytext=center,
                arrowprops=dict(arrowstyle='->', color=C_MIN, lw=2.5))
    ax.text(R * 1.2 * np.cos(original_angle), R * 1.2 * np.sin(original_angle),
            'Original $\\theta_i$', fontsize=9, color=C_MIN, fontweight='bold', ha='left')

    # Multiple perturbed directions
    n_perturbed = 12
    perturbed_angles = original_angle + rng.normal(0, ang_std, n_perturbed)
    for pa in perturbed_angles:
        ax.annotate('', xy=R * 0.95 * np.array([np.cos(pa), np.sin(pa)]),
                    xytext=center,
                    arrowprops=dict(arrowstyle='->', color=C_SYN, lw=0.8, alpha=0.5))

    # Angular std arc
    from matplotlib.patches import Arc
    arc = Arc(center, 1.5, 1.5, angle=0,
              theta1=np.degrees(original_angle - ang_std),
              theta2=np.degrees(original_angle + ang_std),
              color=C_BORDER, linewidth=2.5, linestyle='-')
    ax.add_patch(arc)
    ax.text(0.9 * np.cos(original_angle + ang_std + 0.1),
            0.9 * np.sin(original_angle + ang_std + 0.1),
            f'$\\sigma_{{\\theta}} = {np.degrees(ang_std):.0f}°$',
            fontsize=9, color=C_BORDER, fontweight='bold')

    # Point on the segment
    r_sample = 1.3
    ax.scatter(r_sample * np.cos(original_angle), r_sample * np.sin(original_angle),
               c=C_MIN, s=80, marker='s', edgecolors='black', linewidth=0.6, zorder=5)

    # Perturbed synthetic points
    for pa in perturbed_angles:
        r_syn = r_sample + rng.normal(0, 0.15)
        ax.scatter(r_syn * np.cos(pa), r_syn * np.sin(pa), c=C_SYN, s=30,
                   marker='^', edgecolors='black', linewidth=0.3, alpha=0.7, zorder=4)

    ax.scatter(*center, c='black', marker='+', s=120, linewidths=2, zorder=5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title('Step 5: Angular Noise ($\\theta + \\mathcal{N}(0, \\sigma_\\theta)$)',
                 fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.1)

    handles = [
        Line2D([0], [0], color=C_MIN, linewidth=2.5, label='Original direction'),
        Line2D([0], [0], color=C_SYN, linewidth=0.8, alpha=0.5, label='Perturbed directions'),
        Line2D([0], [0], color=C_BORDER, linewidth=2.5, label=f'$\\pm\\sigma_\\theta$'),
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9, fontsize=8)
    savefig(fig, 'ls_step5_angular_noise.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Gaussian thickening
# ═══════════════════════════════════════════════════════════════════════════
def ls_step6():
    rng = np.random.RandomState(SEED)
    center = np.array([0.0, 0.0])
    angle = np.radians(30)
    r_in, r_out = 0.8, 1.6
    sigma = 0.12

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Draw reference circle
    R = 2.0
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), c=C_DARK, linewidth=0.3, alpha=0.15)

    # Draw segment direction
    direction = np.array([np.cos(angle), np.sin(angle)])
    perp = np.array([-np.sin(angle), np.cos(angle)])

    # Segment line
    p_in = center + r_in * direction
    p_out = center + r_out * direction
    ax.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]], c=C_MIN, linewidth=2.0, zorder=3)

    # Generate points along segment with Gaussian thickening
    n_pts = 40
    r_vals = rng.uniform(r_in, r_out, n_pts)
    perp_offsets = rng.normal(0, sigma, n_pts)
    syn_pts = center + np.outer(r_vals, direction) + np.outer(perp_offsets, perp)

    ax.scatter(syn_pts[:, 0], syn_pts[:, 1], c=C_SYN, s=25, marker='^',
               edgecolors='black', linewidth=0.3, alpha=0.7, zorder=4)

    # Show Gaussian spread at midpoint
    r_mid = (r_in + r_out) / 2
    mid_pt = center + r_mid * direction
    for ns in [1, 2]:
        offset = ns * sigma
        pt_up = mid_pt + offset * perp
        pt_down = mid_pt - offset * perp
        ax.plot([pt_up[0], pt_down[0]], [pt_up[1], pt_down[1]],
                c=C_BORDER, linewidth=1.0 + (2-ns)*0.5, alpha=0.4 + ns*0.2, linestyle='--')
    ax.annotate(f'$\\sigma = {sigma:.2f}$', xy=mid_pt + 2*sigma*perp,
                xytext=mid_pt + 5*sigma*perp + 0.3*direction,
                fontsize=9, color=C_BORDER, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_BORDER, lw=0.8))

    ax.scatter(*center, c='black', marker='+', s=120, linewidths=2, zorder=5)
    ax.set_aspect('equal')
    ax.set_title('Step 6: Gaussian Thickening ($\\mathcal{N}(0, \\sigma)$ perpendicular)',
                 fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.1)

    handles = [
        Line2D([0], [0], color=C_MIN, linewidth=2, label='Segment direction'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_SYN,
               markeredgecolor='black', markersize=7, label='Synthetic (thickened)'),
        Line2D([0], [0], color=C_BORDER, linewidth=1, linestyle='--',
               label=f'$\\pm 1\\sigma, \\pm 2\\sigma$ spread'),
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9, fontsize=8)
    savefig(fig, 'ls_step6_gaussian.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Final generation
# ═══════════════════════════════════════════════════════════════════════════
def ls_step7():
    X_min = make_minority()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))
    n_layers = 5
    boundaries = np.linspace(0, R, n_layers + 1)
    rng = np.random.RandomState(SEED)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Draw layers lightly
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.15, edgecolor=C_DARK,
                      linewidth=0.3, zorder=1)
        ax.add_patch(wedge)

    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=50, alpha=0.8, edgecolors='black',
               linewidth=0.4, marker='s', zorder=3)

    # Generate synthetic points in each layer
    all_syn = []
    for i in range(len(X_min)):
        direction = X_min[i] - center
        angle = np.arctan2(direction[1], direction[0])
        r_pt = np.linalg.norm(direction)
        layer_idx = np.searchsorted(boundaries[1:], r_pt)
        layer_idx = min(layer_idx, n_layers - 1)
        r_in, r_out = boundaries[layer_idx], boundaries[layer_idx + 1]

        n_syn = 2
        for _ in range(n_syn):
            ang_perturbed = angle + rng.normal(0, np.radians(10))
            r_syn = rng.uniform(r_in, r_out)
            perp = np.array([-np.sin(ang_perturbed), np.cos(ang_perturbed)])
            pt = center + r_syn * np.array([np.cos(ang_perturbed), np.sin(ang_perturbed)])
            pt += rng.normal(0, 0.05) * perp
            all_syn.append(pt)

    all_syn = np.array(all_syn)
    ax.scatter(all_syn[:, 0], all_syn[:, 1], c=C_SYN, s=30, marker='^',
               edgecolors='black', linewidth=0.3, alpha=0.8, zorder=4)

    ax.scatter(*center, c='black', marker='+', s=120, linewidths=2, zorder=5)

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN,
               markeredgecolor='black', markersize=8, label='Minority'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_SYN,
               markeredgecolor='black', markersize=8, label=f'Synthetic ($n={len(all_syn)}$)'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)
    ax.set_title('Step 7: Final Synthetic Points (LS-CO)', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    savefig(fig, 'ls_step7_generation.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Variation A: Generic (single center)
# ═══════════════════════════════════════════════════════════════════════════
def ls_var_generic():
    X_min = make_minority()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))
    n_layers = 5
    boundaries = np.linspace(0, R, n_layers + 1)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.35, edgecolor=C_DARK,
                      linewidth=0.6, zorder=1)
        ax.add_patch(wedge)

    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=55, alpha=0.9, edgecolors='black',
               linewidth=0.5, marker='s', zorder=3)
    ax.scatter(*center, c='black', marker='+', s=150, linewidths=2.5, zorder=5)

    # Radial lines to all points
    for pt in X_min:
        ax.plot([center[0], pt[0]], [center[1], pt[1]], c=C_DARK, linewidth=0.3,
                alpha=0.2, zorder=2)

    ax.set_title('Variation A: Generic (Single Center for All)',
                 fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)

    ax.text(0.98, 0.02, 'Single center = mean of all minority\n→ global radial structure',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    savefig(fig, 'ls_var_generic.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Variation B: Cluster-based
# ═══════════════════════════════════════════════════════════════════════════
def ls_var_cluster():
    X_min = make_minority()
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels = km.labels_
    centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for c in range(3):
        mask = labels == c
        pts = X_min[mask]
        center = pts.mean(axis=0)
        R_c = np.max(np.linalg.norm(pts - center, axis=1)) if len(pts) > 1 else 0.3
        n_layers = 4
        boundaries = np.linspace(0, R_c, n_layers + 1)

        for l in range(n_layers):
            r_in, r_out = boundaries[l], boundaries[l+1]
            alpha_val = 0.15 + 0.08 * l
            wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                          facecolor=CLUSTER_COLORS[c], alpha=alpha_val, edgecolor=C_DARK,
                          linewidth=0.3, zorder=1)
            ax.add_patch(wedge)

        ax.scatter(pts[:, 0], pts[:, 1], c=CLUSTER_COLORS[c], s=55, alpha=0.9,
                   edgecolors='black', linewidth=0.5, marker='s', zorder=3)
        ax.scatter(*center, c='black', marker='+', s=100, linewidths=2, zorder=5)
        ax.text(center[0], center[1] + R_c + 0.15, f'Cluster {c+1}',
                fontsize=8, ha='center', fontweight='bold', color=CLUSTER_COLORS[c])

    ax.set_title('Variation B: Cluster-Based (Per-Cluster Layers)',
                 fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)

    ax.text(0.98, 0.02, 'Each cluster gets its own center + layers\n→ preserves cluster structure',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    savefig(fig, 'ls_var_cluster.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Overview: 2x2 panel
# ═══════════════════════════════════════════════════════════════════════════
def ls_overview():
    X_min = make_minority()
    center = X_min.mean(axis=0)
    R = np.max(np.linalg.norm(X_min - center, axis=1))
    n_layers = 5
    boundaries = np.linspace(0, R, n_layers + 1)
    rng = np.random.RandomState(SEED)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # (a) Layers
    ax = axes[0, 0]
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.4, edgecolor=C_DARK,
                      linewidth=0.5, zorder=1)
        ax.add_patch(wedge)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, edgecolors='black',
               linewidth=0.3, marker='s', zorder=3)
    ax.scatter(*center, c='black', marker='+', s=100, linewidths=2, zorder=5)
    ax.set_title('(a) Concentric Layers', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    # (b) Segments
    ax = axes[0, 1]
    for l in range(n_layers):
        r_in, r_out = boundaries[l], boundaries[l+1]
        wedge = Wedge(center, r_out, 0, 360, width=r_out - r_in,
                      facecolor=LAYER_COLORS[l], alpha=0.15, edgecolor=C_DARK,
                      linewidth=0.3, zorder=1)
        ax.add_patch(wedge)
    for pt in X_min:
        ax.plot([center[0], pt[0]], [center[1], pt[1]], c=C_DARK, lw=0.4, alpha=0.3)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, edgecolors='black',
               linewidth=0.3, marker='s', zorder=3)
    ax.scatter(*center, c='black', marker='+', s=100, linewidths=2, zorder=5)
    ax.set_title('(b) Radial Segments', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    # (c) Allocation bar chart
    ax = axes[1, 0]
    dists = np.linalg.norm(X_min - center, axis=1)
    counts = [np.sum((dists >= boundaries[l]) & (dists < boundaries[l+1])) for l in range(n_layers)]
    areas = [np.pi * (boundaries[l+1]**2 - boundaries[l]**2) for l in range(n_layers)]
    area_arr = np.array(areas)
    alloc = (area_arr / area_arr.sum() * 30).astype(int)
    x_pos = np.arange(n_layers)
    ax.bar(x_pos - 0.2, counts, 0.35, color=C_MIN, alpha=0.7, label='Points')
    ax.bar(x_pos + 0.2, alloc, 0.35, color=C_SYN, alpha=0.7, label='Allocated')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$L_{l+1}$' for l in range(n_layers)])
    ax.set_title('(c) Layer Allocation', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.15, axis='y')

    # (d) Generation
    ax = axes[1, 1]
    for l in range(n_layers):
        wedge = Wedge(center, boundaries[l+1], 0, 360, width=boundaries[l+1]-boundaries[l],
                      facecolor=LAYER_COLORS[l], alpha=0.12, edgecolor=C_DARK,
                      linewidth=0.2, zorder=1)
        ax.add_patch(wedge)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, edgecolors='black',
               linewidth=0.3, marker='s', zorder=3)
    syn = []
    for pt in X_min:
        d = X_min[0:1] - center  # simplified
        a = np.arctan2(*(pt - center)[::-1])
        for _ in range(2):
            r_s = np.linalg.norm(pt - center) + rng.normal(0, 0.1)
            a_s = a + rng.normal(0, np.radians(10))
            syn.append(center + r_s * np.array([np.cos(a_s), np.sin(a_s)]))
    syn = np.array(syn)
    ax.scatter(syn[:, 0], syn[:, 1], c=C_SYN, s=25, marker='^',
               edgecolors='black', linewidth=0.3, alpha=0.7, zorder=4)
    ax.scatter(*center, c='black', marker='+', s=100, linewidths=2, zorder=5)
    ax.set_title('(d) Final Generation', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    fig.suptitle('LS-CO: Layered Segmental Circular Oversampling — Overview',
                 fontweight='bold', fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, 'ls_overview.pdf')


if __name__ == '__main__':
    print('Generating LS-CO step-by-step figures...')
    ls_step1()
    ls_step2()
    ls_step3()
    ls_step4()
    ls_step5()
    ls_step6()
    ls_step7()
    ls_var_generic()
    ls_var_cluster()
    ls_overview()
    print('Done! 10 LS-CO figures generated.')
