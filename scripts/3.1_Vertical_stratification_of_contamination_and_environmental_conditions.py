import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import PchipInterpolator


# Global style
# =========================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

# Final size
FIG_W = 3.5
FIG_H = 3.8
DPI = 600

# Colors
COLOR_LIGHT = '#589C4F'   # C6-C9
COLOR_HEAVY = '#D15240'   # C10-C40
COLOR_PY = '#C62F22'      # Puyang
COLOR_HZ = '#0F77B5'      # Hangzhou
COLOR_LOAD = '#7A1F5C'    
COLOR_AXIS = '#333333'
COLOR_GRID = '#D9D9D9'

# Font sizes
FS_PANEL = 8.5
FS_LABEL = 7.6
FS_TICK = 6.6
FS_LEGEND = 6.2
FS_TEXT = 6.0


# Load data
# =========================
sample_info = pd.read_csv('1_Sample_metadata_and_inclusion_status.txt', sep='\t')
py_env = pd.read_csv('2_Environmental_factors_and_pollutant_profiles_Puyang.txt', sep='\t')
hz_env = pd.read_csv('2_Environmental_factors_and_pollutant_profiles_Hangzhou.txt', sep='\t')

used = sample_info[
    sample_info['是否参与 MAG abundance 分析'].astype(str).str.lower() == 'yes'
].copy()
used['site_norm'] = used['site'].astype(str).str.strip().str.lower()
used['depth_mid'] = pd.to_numeric(used['depth'], errors='coerce')


# Panels a/b: profiles
# =====================
py = used[used['site_norm'] == 'puyang'][['sample_ID', 'depth_mid']].merge(
    py_env, left_on='sample_ID', right_on='points', how='left'
)
py['C6-C9'] = pd.to_numeric(py['C6_C9'], errors='coerce')
py['C10-C40'] = pd.to_numeric(py['C10_C40'], errors='coerce')
py = py[['sample_ID', 'depth_mid', 'C6-C9', 'C10-C40']].copy()

hz = used[used['site_norm'] == 'hangzhou'][['sample_ID', 'depth_mid']].merge(
    hz_env, left_on='sample_ID', right_on='sample_id', how='left'
)
hz['C6-C9'] = pd.to_numeric(hz['TPH (C6-C9)'], errors='coerce')
hz['C10-C40'] = pd.to_numeric(hz['TPH (C10-C40)'], errors='coerce')
hz = hz[['sample_ID', 'depth_mid', 'C6-C9', 'C10-C40']].copy()

def positive_floor(series):
    pos = series[series > 0]
    return float(pos.min()) / 2 if len(pos) else 1e-3

def max_by_depth(df):
    out = (
        df.groupby('depth_mid', as_index=False)
        .agg({'C6-C9': 'max', 'C10-C40': 'max'})
        .sort_values('depth_mid')
        .reset_index(drop=True)
    )
    return out

py_prof = max_by_depth(py)
hz_prof = max_by_depth(hz)

py_c69_floor = positive_floor(py_prof['C6-C9'])
py_c1040_floor = positive_floor(py_prof['C10-C40'])
hz_c69_floor = positive_floor(hz_prof['C6-C9'])
hz_c1040_floor = positive_floor(hz_prof['C10-C40'])

py_prof['C6-C9_plot'] = py_prof['C6-C9'].where(py_prof['C6-C9'] > 0, py_c69_floor)
py_prof['C10-C40_plot'] = py_prof['C10-C40'].where(py_prof['C10-C40'] > 0, py_c1040_floor)
hz_prof['C6-C9_plot'] = hz_prof['C6-C9'].where(hz_prof['C6-C9'] > 0, hz_c69_floor)
hz_prof['C10-C40_plot'] = hz_prof['C10-C40'].where(hz_prof['C10-C40'] > 0, hz_c1040_floor)

all_x = pd.concat([
    py_prof['C6-C9_plot'], py_prof['C10-C40_plot'],
    hz_prof['C6-C9_plot'], hz_prof['C10-C40_plot']
], ignore_index=True)
all_x = all_x.replace([np.inf, -np.inf], np.nan).dropna()
xmin = 10 ** np.floor(np.log10(all_x.min()))
xmax = 10 ** np.ceil(np.log10(all_x.max()))

def smooth_profile_xy(x, y, n=250):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    x = x[order]
    y = y[order]
    if len(x) < 3:
        return x, y
    interp = PchipInterpolator(y, x)
    y_new = np.linspace(y.min(), y.max(), n)
    x_new = interp(y_new)
    return x_new, y_new

def draw_gradient_band_left_of_curve(ax, x_curve, y_curve, color,
                                     band_frac=0.12, n_bands=28, alpha_max=0.5):
    logx = np.log10(x_curve)
    width = band_frac * (np.log10(xmax) - np.log10(xmin))
    for i in range(n_bands):
        frac0 = i / n_bands
        frac1 = (i + 1) / n_bands
        left0 = 10 ** (logx - width * frac0)
        left1 = 10 ** (logx - width * frac1)
        ax.fill_betweenx(
            y_curve, left1, left0,
            color=color,
            alpha=alpha_max * (1 - frac0) ** 0.35,
            linewidth=0,
            zorder=1
        )

def add_profile_with_shading(ax, df, xcol, ycol, color, label):
    x = df[xcol].values
    y = df[ycol].values
    xs, ys = smooth_profile_xy(x, y, n=250)
    draw_gradient_band_left_of_curve(ax, xs, ys, color=color,
                                     band_frac=0.12, n_bands=28, alpha_max=0.18)
    ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.95, zorder=2)
    ax.scatter(
        x, y,
        marker='s',
        s=20,
        facecolor=color,
        edgecolor='black',
        linewidth=0.4,
        label=label,
        zorder=3
    )


# PCA data
# ==================
py_env2 = used[used['site_norm'] == 'puyang'][['sample_ID', 'depth_mid']].merge(
    py_env, left_on='sample_ID', right_on='points', how='left'
).copy()
hz_env2 = used[used['site_norm'] == 'hangzhou'][['sample_ID', 'depth_mid']].merge(
    hz_env, left_on='sample_ID', right_on='sample_id', how='left'
).copy()

py_env2 = py_env2.rename(columns={
    'water': 'moisture',
    '全盐量': 'salinity',
    '电导率': 'conduc',
    '有机碳': 'OC',
    '总氮': 'TN',
    '硝氮': 'NO3-N',
    '铵氮': 'NH4-N',
    '总磷': 'TP',
    '有效磷': 'availP',
    '阳离子交换量': 'CEC',
    '氟': 'F',
    '镍': 'Ni',
    '铅': 'Pb',
    '砷': 'As',
    '镉': 'Cd',
    '铝': 'Al',
    '铁': 'Fe',
    '锰': 'Mn',
    '钒': 'V',
    'C6_C9': 'C6-C9',
    'C10_C40': 'C10-C40'
})

hz_env2 = hz_env2.rename(columns={
    '含水率': 'moisture',
    '全盐量': 'salinity',
    '电导率': 'conduc',
    '有机碳': 'OC',
    '总氮': 'TN',
    '硝氮': 'NO3-N',
    '铵氮': 'NH4-N',
    '总磷': 'TP',
    'availP': 'availP',
    'CEC': 'CEC',
    '氟': 'F',
    '镍': 'Ni',
    '铅': 'Pb',
    '砷': 'As',
    '镉': 'Cd',
    '铝': 'Al',
    '铁': 'Fe',
    '锰': 'Mn',
    '钒': 'V',
    'TPH (C6-C9)': 'C6-C9',
    'TPH (C10-C40)': 'C10-C40'
})

shared_vars = [
    'pH', 'moisture', 'salinity', 'conduc', 'OC', 'TN', 'NO3-N', 'NH4-N',
    'TP', 'availP', 'CEC', 'F', 'Ni', 'Pb', 'As', 'Cd', 'Al', 'Fe', 'Mn',
    'V', 'C6-C9', 'C10-C40'
]

py_pca = py_env2[['sample_ID', 'depth_mid'] + [c for c in shared_vars if c in py_env2.columns]].copy()
py_pca['site'] = 'Puyang'

hz_pca = hz_env2[['sample_ID', 'depth_mid'] + [c for c in shared_vars if c in hz_env2.columns]].copy()
hz_pca['site'] = 'Hangzhou'

env_df = pd.concat([py_pca, hz_pca], ignore_index=True)

for col in shared_vars:
    env_df[col] = pd.to_numeric(env_df[col], errors='coerce')

X = env_df[shared_vars].apply(lambda s: s.fillna(s.median()), axis=0)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100

env_df['PC1'] = scores[:, 0]
env_df['PC2'] = scores[:, 1]

loadings = pd.DataFrame(
    pca.components_.T,
    index=shared_vars,
    columns=['PC1', 'PC2']
)
loadings['mag'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)

def add_confidence_ellipse(ax, x, y, color, n_std=1.6, alpha=0.10, lw=0.9):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if np.linalg.det(cov) <= 0:
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        lw=lw,
        zorder=1
    )
    ax.add_patch(ell)


# Plotting
# ==================
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.0, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.42,
    hspace=0.35
)

ax1 = fig.add_subplot(gs[0, 0], box_aspect=1)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1, box_aspect=1)
ax3 = fig.add_subplot(gs[1, 0], box_aspect=1)
ax4 = fig.add_subplot(gs[1, 1], box_aspect=1)

# a
add_profile_with_shading(ax1, py_prof, 'C6-C9_plot', 'depth_mid', COLOR_LIGHT, 'C6–C9')
add_profile_with_shading(ax1, py_prof, 'C10-C40_plot', 'depth_mid', COLOR_HEAVY, 'C10–C40')
ax1.set_xscale('log')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(5.5, 0)
ax1.set_ylabel('Depth (m)', fontsize=FS_LABEL, color=COLOR_AXIS)
ax1.text(0.01, 1.02, 'a', transform=ax1.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')

# b
add_profile_with_shading(ax2, hz_prof, 'C6-C9_plot', 'depth_mid', COLOR_LIGHT, 'C6–C9')
add_profile_with_shading(ax2, hz_prof, 'C10-C40_plot', 'depth_mid', COLOR_HEAVY, 'C10–C40')
ax2.set_xscale('log')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(5.5, 0)
ax2.set_ylabel('')
ax2.tick_params(axis='y', labelleft=False)
ax2.legend(frameon=False, fontsize=FS_LEGEND, loc='lower right', handletextpad=0.5, borderpad=0.2)
ax2.text(0.01, 1.02, 'b', transform=ax2.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')

for ax in [ax1, ax2]:
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.6)
    ax.tick_params(axis='both', labelsize=FS_TICK, colors=COLOR_AXIS, length=3.0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(COLOR_AXIS)

# c
for site_name, color in [('Puyang', COLOR_PY), ('Hangzhou', COLOR_HZ)]:
    sub = env_df[env_df['site'] == site_name]
    add_confidence_ellipse(ax3, sub['PC1'], sub['PC2'], color=color)
    ax3.scatter(
        sub['PC1'], sub['PC2'],
        s=26, color=color, edgecolor='white', linewidth=0.4,
        label=site_name, zorder=3
    )

ax3.axhline(0, linewidth=0.7, color='#A6A6A6', linestyle='--', zorder=0)
ax3.axvline(0, linewidth=0.7, color='#A6A6A6', linestyle='--', zorder=0)
ax3.set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=FS_LABEL, color=COLOR_AXIS)
ax3.set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=FS_LABEL, color=COLOR_AXIS)
ax3.legend(frameon=False, fontsize=FS_LEGEND, loc='upper right', handletextpad=0.4, borderpad=0.2)
ax3.tick_params(axis='both', labelsize=FS_TICK, colors=COLOR_AXIS, length=3.0, pad=2)
ax3.text(0.01, 1.02, 'c', transform=ax3.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')
for sp in ax3.spines.values():
    sp.set_color(COLOR_AXIS)

# d: unified loading arrows
top_vars = loadings.sort_values('mag', ascending=False).head(8)

scale = 3.0
for var, row in top_vars.iterrows():
    x = row['PC1'] * scale
    y = row['PC2'] * scale
    ax4.arrow(
        0, 0, x, y,
        color=COLOR_LOAD, linewidth=1.0,
        head_width=0.08, head_length=0.11,
        length_includes_head=True, zorder=2
    )
    ax4.text(
        x + (0.06 if x >= 0 else -0.06),
        y + (0.05 if y >= 0 else -0.05),
        var,
        fontsize=FS_TEXT,
        color=COLOR_LOAD,
        ha='left' if x >= 0 else 'right',
        va='center'
    )

ax4.axhline(0, linewidth=0.7, color='#A6A6A6', linestyle='--', zorder=0)
ax4.axvline(0, linewidth=0.7, color='#A6A6A6', linestyle='--', zorder=0)
ax4.set_xlim(None, 1.3)
ax4.set_ylim(-0.7, 1.5)
ax4.set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=FS_LABEL, color=COLOR_AXIS)
ax4.set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=FS_LABEL, color=COLOR_AXIS)
ax4.tick_params(axis='both', labelsize=FS_TICK, colors=COLOR_AXIS, length=3.0, pad=2)
ax4.text(0.01, 1.02, 'd', transform=ax4.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')
for sp in ax4.spines.values():
    sp.set_color(COLOR_AXIS)

# Shared x label for a and b
fig.canvas.draw()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
x_center_ab = (pos1.x0 + pos2.x1) / 2
y_bottom_ab = min(pos1.y0, pos2.y0)
fig.text(
    x_center_ab, y_bottom_ab - 0.055,
    'Hydrocarbon concentration in soil (mg/kg)',
    ha='center', va='top', fontsize=FS_LABEL, color=COLOR_AXIS
)


out_png = Path('Figure1_4panel.png')
out_pdf = Path('Figure1_4panel.pdf')
out_tif = Path('Figure1_4panel.tif')

fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_tif, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'Saved PNG: {out_png}')
print(f'Saved PDF: {out_pdf}')
print(f'Saved TIFF: {out_tif}')
