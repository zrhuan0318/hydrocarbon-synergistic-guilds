import os

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# 全局设置
# -----------------------------
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'axes.linewidth': 0.7})
FIGSIZE = (2.8, 2.0)
DPI = 600
FS_AXIS, FS_TICK, FS_LEGEND = 7.8, 6.4, 6.0
site_colors = {'Puyang': '#E69F00', 'Hangzhou': '#56B4E9'}

# -----------------------------
# 数据读取与绘图
# -----------------------------
files = {
    'Puyang C10-C40': 'Puyang_C10C40_sid_ready_sid_pairwise_network_annotated.tsv',
    'Hangzhou C10-C40': 'Hangzhou_C10C40_sid_ready_sid_pairwise_network_annotated.tsv',
    'Puyang C6-C9': 'Puyang_C6C9_sid_ready_sid_pairwise_network_annotated.tsv',
    'Hangzhou C6-C9': 'Hangzhou_C6C9_sid_ready_sid_pairwise_network_annotated.tsv'
}

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

for label, fname in files.items():
    if not Path(fname).exists(): continue

    df = pd.read_csv(fname, sep='\t')
    df = df.sort_values('synergy', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    site = 'Puyang' if 'Puyang' in label else 'Hangzhou'
    linestyle = '-' if 'C10-C40' in label else '--'

    ax.plot(df['rank'], df['synergy'], lw=1.3, color=site_colors[site],
            linestyle=linestyle, alpha=0.9)

# 坐标轴修饰
ax.set_xlabel('Rank of synergistic taxon pairs', fontsize=FS_AXIS)
ax.set_ylabel('Pairwise synergy value', fontsize=FS_AXIS)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='-', linewidth=0.3, alpha=0.2)

# 手动自定义图例
handles = [
    mpl.lines.Line2D([0], [0], color=site_colors['Puyang'], lw=1.4, label='PY C10–C40'),
    mpl.lines.Line2D([0], [0], color=site_colors['Hangzhou'], lw=1.4, label='HZ C10–C40'),
    mpl.lines.Line2D([0], [0], color=site_colors['Puyang'], lw=1.4, ls='--', label='PY C6–C9'),
    mpl.lines.Line2D([0], [0], color=site_colors['Hangzhou'], lw=1.4, ls='--', label='HZ C6–C9'),
]
ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc='upper right')

plt.savefig('Figure4e_Line.pdf', bbox_inches='tight')
print("Generated: Figure4e_Line.pdf")