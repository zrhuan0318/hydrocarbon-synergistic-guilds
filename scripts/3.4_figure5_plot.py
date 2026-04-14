import os
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Global style
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# 1. Input files
# =========================
py_mat = pd.read_csv("Puyang_soft_stage_coupling_weighted_matrix.csv", index_col=0)
hz_mat = pd.read_csv("Hangzhou_soft_stage_coupling_weighted_matrix.csv", index_col=0)

py_comp = pd.read_csv("Puyang_soft_stage_coupling_composition.csv")
hz_comp = pd.read_csv("Hangzhou_soft_stage_coupling_composition.csv")

py_depth = pd.read_csv("Puyang_depth_resolved_stage_succession_soft.csv")
hz_depth = pd.read_csv("Hangzhou_depth_resolved_stage_succession_soft.csv")

py_sci = pd.read_csv("Puyang_soft_stage_coupling_index.csv")
hz_sci = pd.read_csv("Hangzhou_soft_stage_coupling_index.csv")

# =========================
# 2. Harmonize order
# =========================
stage_order = ["Stage I", "Stage II", "Stage III"]
depth_order = ["0-1.5 m", "1.5-3.0 m", "3.0-5.0 m"]
major_pairs = ["Stage II -> Stage II", "Stage II -> Stage III", "Stage III -> Stage II", "Stage III -> Stage III"]

py_mat = py_mat.reindex(index=stage_order, columns=stage_order)
hz_mat = hz_mat.reindex(index=stage_order, columns=stage_order)

py_comp = py_comp.set_index("stage_pair").reindex(major_pairs).reset_index()
hz_comp = hz_comp.set_index("stage_pair").reindex(major_pairs).reset_index()

py_depth["depth_bin"] = pd.Categorical(py_depth["depth_bin"], categories=depth_order, ordered=True)
hz_depth["depth_bin"] = pd.Categorical(hz_depth["depth_bin"], categories=depth_order, ordered=True)
py_depth = py_depth.sort_values("depth_bin")
hz_depth = hz_depth.sort_values("depth_bin")

soft_sci_py = float(py_sci["soft_stage_coupling_index"].iloc[0])
soft_sci_hz = float(hz_sci["soft_stage_coupling_index"].iloc[0])

# =========================
# 3. Colors
# =========================
# d panel only
color_py = "#FFA453"  #"#4C78A8"
color_hz = "#4EA8C0"  #"#E45756"

# c panel unified stage colors
stage_colors = {
    "Stage I_soft": "#72B7B2",
    "Stage II_soft": "#F2CF5B",
    "Stage III_soft": "#B279A2",
}

# =========================
# 4. Helpers
# =========================
def bubble_size(val, vmax, size_scale=3200):
    if val <= 0 or vmax <= 0:
        return 0
    return (val / vmax) * size_scale

def add_panel_letter(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=9)

def bubble_matrix(ax, mat, vmax_global, size_scale=3200, show_ylabel=True, show_yticklabels=True):
    x_labels = list(mat.columns)
    y_labels = list(mat.index)

    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            val = mat.loc[y, x]
            size = bubble_size(val, vmax_global, size_scale=size_scale)
            ax.scatter(j, i, s=size, alpha=0.75)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=7)

    if show_yticklabels:
        ax.set_yticklabels(y_labels, fontsize=7)
    else:
        ax.set_yticklabels([])

    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(len(y_labels) - 0.5, -0.5)
    ax.set_xlabel("Target-side stage")

    if show_ylabel:
        ax.set_ylabel("Source-side stage")
    else:
        ax.set_ylabel("")

    for x in np.arange(-0.5, len(x_labels), 1):
        ax.axvline(x, linewidth=0.5, alpha=0.25, zorder=0)
    for y in np.arange(-0.5, len(y_labels), 1):
        ax.axhline(y, linewidth=0.5, alpha=0.25, zorder=0)

def add_bubble_size_legend(ax, vmax_global, size_scale=3200):
    legend_vals = [0.25 * vmax_global, 0.50 * vmax_global, 1.00 * vmax_global]
    legend_sizes = [bubble_size(v, vmax_global, size_scale=size_scale) for v in legend_vals]
    handles = [ax.scatter([], [], s=s, alpha=0.75) for s in legend_sizes]
    labels = [f"{v:.2f}" for v in legend_vals]
    ax.legend(handles, labels, title="Weighted synergy",
              frameon=False, scatterpoints=1,
              loc="upper left", bbox_to_anchor=(1.02, 1.02),
              title_fontsize=7, fontsize=7)

def stacked_stage_panel(ax, py_depth, hz_depth):
    py_stage = py_depth.set_index("depth_bin")[["Stage I_soft", "Stage II_soft", "Stage III_soft"]].loc[depth_order]
    hz_stage = hz_depth.set_index("depth_bin")[["Stage I_soft", "Stage II_soft", "Stage III_soft"]].loc[depth_order]

    x_py = np.array([0, 1, 2])
    x_hz = np.array([4, 5, 6])
    width = 0.75

    order_cols = ["Stage I_soft", "Stage II_soft", "Stage III_soft"]
    order_labels = ["Stage I", "Stage II", "Stage III"]

    # Puyang
    bottom = np.zeros(len(x_py))
    for col, label in zip(order_cols, order_labels):
        vals = py_stage[col].values
        ax.bar(x_py, vals, width=width, bottom=bottom, color=stage_colors[col], label=label)
        bottom += vals

    # Hangzhou (use SAME colors / legend)
    bottom = np.zeros(len(x_hz))
    for col in order_cols:
        vals = hz_stage[col].values
        ax.bar(x_hz, vals, width=width, bottom=bottom, color=stage_colors[col])
        bottom += vals

    ax.set_xticks(list(x_py) + list(x_hz))
    ax.set_xticklabels(
        ["0-1.5", "1.5-3.0", "3.0-5.0", "0-1.5", "1.5-3.0", "3.0-5.0"],
        fontsize=7
    )
    ax.set_xlabel("Soil depth (m)")
    ax.set_ylabel("Stage proportion")
    ax.set_ylim(0, 1.02)
    ax.axvline(3.0, linewidth=0.8, linestyle="--", alpha=0.5)

    # site labels above the plot
    ax.text(1, 1.04, "Puyang", ha="center", va="bottom", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.text(5, 1.04, "Hangzhou", ha="center", va="bottom", fontsize=8,
            transform=ax.get_xaxis_transform())

    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))

def paired_lollipop(ax, py_comp, hz_comp, sci_py, sci_hz):
    y = np.arange(len(major_pairs))

    py_vals = py_comp["weighted_proportion"].values
    hz_vals = hz_comp["weighted_proportion"].values

    for i in range(len(y)):
        ax.plot([py_vals[i], hz_vals[i]], [y[i], y[i]], linewidth=1.0, alpha=0.6, color="black")
        ax.scatter(py_vals[i], y[i], s=40, facecolors=color_py, edgecolors=color_py,
                   label=None, zorder=3)
        ax.scatter(hz_vals[i], y[i], s=40, marker="s", facecolors=color_hz, edgecolors=color_hz,
                   label=None, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(major_pairs, fontsize=7)
    ax.set_xlabel("Weighted proportion")
    ax.invert_yaxis()

    xmax = max(py_vals.max(), hz_vals.max()) * 1.45
    ax.set_xlim(0, xmax)

    # legend inside panel, includes SCI
    h1 = ax.scatter([], [], s=40, facecolors=color_py, edgecolors=color_py)
    h2 = ax.scatter([], [], s=40, marker="s", facecolors=color_hz, edgecolors=color_hz)
    labels = [f"Puyang, SCI={sci_py:.3f}", f"Hangzhou, SCI={sci_hz:.3f}"]
    ax.legend([h1, h2], labels, frameon=False, fontsize=7,
              loc="lower right", handletextpad=0.6, borderaxespad=0.4)

# =========================
# 5. Plot
# =========================
fig = plt.figure(figsize=(7.1, 5.6), dpi=300, constrained_layout=True)

# a top-left; c top-right; b bottom-left; d bottom-right
# keep a/c slightly apart; leave right margin for legends
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.0, 0.92],
    height_ratios=[1, 1.05],
    wspace=0.06, hspace=0.14
)

ax1 = fig.add_subplot(gs[0, 0])  # a
ax3 = fig.add_subplot(gs[0, 1])  # c (position swapped)
ax2 = fig.add_subplot(gs[1, 0])  # b (position swapped)
ax4 = fig.add_subplot(gs[1, 1])  # d

vmax_ab = max(py_mat.to_numpy().max(), hz_mat.to_numpy().max())

# a unchanged
bubble_matrix(ax1, py_mat, vmax_global=vmax_ab, size_scale=3200,
              show_ylabel=True, show_yticklabels=True)

# b now in lower-left, restore y-axis label and ticks
bubble_matrix(ax2, hz_mat, vmax_global=vmax_ab, size_scale=3200,
              show_ylabel=True, show_yticklabels=True)

# put bubble legend in the empty right side of b
# add_bubble_size_legend(ax2, vmax_global=vmax_ab, size_scale=3200)

stacked_stage_panel(ax3, py_depth, hz_depth)
paired_lollipop(ax4, py_comp, hz_comp, soft_sci_py, soft_sci_hz)

add_panel_letter(ax1, "a")
add_panel_letter(ax2, "b")
add_panel_letter(ax3, "c")
add_panel_letter(ax4, "d")

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(labelsize=7)

# =========================
# 6. Save
# =========================
out = Path("Figure5_soft_stage_coupling_bubble_refined_v4.png")
fig.savefig(out, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved to: {out}")