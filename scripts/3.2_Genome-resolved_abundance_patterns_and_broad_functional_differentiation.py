import os
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
from matplotlib.patches import Ellipse

# -----------------------------
# Global style for main-text figure
# -----------------------------
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# Main-text figure size / resolution
# Slightly widened to give more room for all x tick labels in panels a/b
FIG_W = 7.6   # inches
FIG_H = 4.8   # inches
DPI = 600

# Unified font sizes
FS_AXIS = 7.8
FS_TICK = 6.4
FS_TICK_X = 5.2
FS_TICK_Y = 6.2
FS_CBAR = 6.6
FS_LEGEND = 6.8
FS_PANEL = 9.0

# Unified site colors for panels c and d
COLOR_PY = '#E69F00'   # '#FFA453'
COLOR_HZ = '#56B4E9'   # '#4EA8C0'

# -----------------------------
# Load data
# -----------------------------
sample = pd.read_csv('1_样品信息表.txt', sep='\t')
py_ab = pd.read_csv('3_濮阳MAG丰度表.txt', sep='\t')
hz_ab = pd.read_csv('4_杭州MAG丰度表.txt', sep='\t')
py_ann = pd.read_csv('5_濮阳MAG注释表.txt', sep='\t').rename(columns={'MAG ID': 'MAG_ID'})
hz_ann = pd.read_csv('5_杭州MAG注释表.txt', sep='\t')
py_fun = pd.read_csv('6_濮阳MAG功能注释表.txt', sep='\t', low_memory=False)
hz_fun = pd.read_csv('6_杭州MAG功能注释表.txt', sep='\t', low_memory=False)
py_env = pd.read_csv('2_濮阳环境因子与污染物表.txt', sep='\t')
hz_env = pd.read_csv('2_杭州环境因子与污染物表.txt', sep='\t')

# -----------------------------
# Metadata
# -----------------------------
used = sample[sample['是否参与 MAG abundance 分析'].astype(str).str.lower() == 'yes'].copy()
used['site_norm'] = used['site'].astype(str).str.strip().str.lower()
used['depth'] = pd.to_numeric(used['depth'], errors='coerce')

py_meta = used[used['site_norm'] == 'puyang'][['sample_ID', 'depth']].merge(
    py_env, left_on='sample_ID', right_on='points', how='left'
)
py_meta['C6-C9'] = pd.to_numeric(py_meta['C6_C9'], errors='coerce')
py_meta['C10-C40'] = pd.to_numeric(py_meta['C10_C40'], errors='coerce')
py_meta = py_meta[['sample_ID', 'depth', 'C6-C9', 'C10-C40']].copy()

hz_meta = used[used['site_norm'] == 'hangzhou'][['sample_ID', 'depth']].merge(
    hz_env, left_on='sample_ID', right_on='sample_id', how='left'
)
hz_meta['C6-C9'] = pd.to_numeric(hz_meta['TPH (C6-C9)'], errors='coerce')
hz_meta['C10-C40'] = pd.to_numeric(hz_meta['TPH (C10-C40)'], errors='coerce')
hz_meta = hz_meta[['sample_ID', 'depth', 'C6-C9', 'C10-C40']].copy()

# -----------------------------
# Abundance processing
# -----------------------------
def prep_abundance(ab):
    ab = ab.copy()
    ab = ab.rename(columns={ab.columns[0]: 'MAG_ID'})
    return ab

py_ab2 = prep_abundance(py_ab)
hz_ab2 = prep_abundance(hz_ab)

# Use intersecting samples to avoid sample-ID mismatch
py_samples = [c for c in py_ab2.columns if c != 'MAG_ID']
hz_samples = [c for c in hz_ab2.columns if c != 'MAG_ID']
py_common = [s for s in py_meta['sample_ID'].tolist() if s in py_samples]
hz_common = [s for s in hz_meta['sample_ID'].tolist() if s in hz_samples]

py_meta2 = py_meta[py_meta['sample_ID'].isin(py_common)].copy()
hz_meta2 = hz_meta[hz_meta['sample_ID'].isin(hz_common)].copy()
py_ab3 = py_ab2[['MAG_ID'] + py_common].copy()
hz_ab3 = hz_ab2[['MAG_ID'] + hz_common].copy()

def abundance_long(ab, site):
    return ab.melt(id_vars='MAG_ID', var_name='sample_ID', value_name='abundance').assign(site=site)

py_long = abundance_long(py_ab3, 'Puyang')
hz_long = abundance_long(hz_ab3, 'Hangzhou')

def mag_correlations(long_df, meta_df):
    df = long_df.merge(meta_df, on='sample_ID', how='left')
    out = []
    for mag, sub in df.groupby('MAG_ID'):
        prevalence = (sub['abundance'] > 0).mean()
        mean_abundance = sub['abundance'].mean()
        for var in ['C6-C9', 'C10-C40', 'depth']:
            rho, p = stats.spearmanr(sub['abundance'], sub[var], nan_policy='omit')
            out.append([mag, var, rho, p, prevalence, mean_abundance])
    return pd.DataFrame(out, columns=['MAG_ID', 'variable', 'rho', 'p', 'prevalence', 'mean_abundance'])

py_cor = mag_correlations(py_long, py_meta2)
hz_cor = mag_correlations(hz_long, hz_meta2)

def best_associations(cor_df, ann_df, site_name):
    sig = cor_df[
        cor_df['variable'].isin(['C6-C9', 'C10-C40']) &
        (cor_df['p'] < 0.05) &
        (cor_df['prevalence'] >= 0.20)
    ].copy()
    best = sig.sort_values(['MAG_ID', 'rho'], ascending=[True, False]).groupby('MAG_ID').head(1).copy()
    best = best.merge(
        ann_df[['MAG_ID', 'Phylum', 'Genus', 'Completeness', 'Contamination']],
        on='MAG_ID', how='left'
    )
    best['site'] = site_name
    return best.sort_values('rho', ascending=False)

py_best = best_associations(py_cor, py_ann, 'Puyang')
hz_best = best_associations(hz_cor, hz_ann, 'Hangzhou')

# -----------------------------
# KO matrices for PCA
# -----------------------------
def ko_presence(fun_df):
    tmp = fun_df.copy().rename(columns={fun_df.columns[0]: 'KO'})
    for col in tmp.columns[1:]:
        tmp[col] = pd.to_numeric(tmp[col], errors='coerce').fillna(0)
    pa = (tmp.set_index('KO').T > 0).astype(int)
    pa.index.name = 'MAG_ID'
    return pa

py_pa = ko_presence(py_fun)
hz_pa = ko_presence(hz_fun)

# -----------------------------
# Top MAGs for display
# -----------------------------
n_top = 15
py_top = py_best.head(n_top).copy()
hz_top = hz_best.head(n_top).copy()

py_top['label'] = py_top.apply(
    lambda r: f"{r['MAG_ID']} | {r['Phylum']}" if pd.isna(r['Genus']) else f"{r['MAG_ID']} | {r['Genus']}",
    axis=1
)
hz_top['label'] = hz_top.apply(
    lambda r: f"{r['MAG_ID']} | {r['Phylum']}" if pd.isna(r['Genus']) else f"{r['MAG_ID']} | {r['Genus']}",
    axis=1
)

# Keep the original ordering logic unchanged:
# first by depth, then by C10-C40 within the same depth
py_order = py_meta2.sort_values(['depth', 'C10-C40'], ascending=[True, False])['sample_ID'].tolist()
hz_order = hz_meta2.sort_values(['depth', 'C10-C40'], ascending=[True, False])['sample_ID'].tolist()

py_heat = py_ab3.set_index('MAG_ID').loc[py_top['MAG_ID'], py_order]
hz_heat = hz_ab3.set_index('MAG_ID').loc[hz_top['MAG_ID'], hz_order]

py_heat_plot = np.log10(py_heat + 1e-6)
hz_heat_plot = np.log10(hz_heat + 1e-6)

# Shared heatmap scale so a/b can use one unified colorbar
heat_vmin = min(py_heat_plot.min().min(), hz_heat_plot.min().min())
heat_vmax = max(py_heat_plot.max().max(), hz_heat_plot.max().max())

# -----------------------------
# Phylum composition
# -----------------------------
def phylum_comp(best_df, top_n=7):
    counts = best_df['Phylum'].fillna('Unclassified').value_counts()
    top = counts.head(top_n).index.tolist()
    out = counts.reindex(top).fillna(0).astype(int)
    other = counts.iloc[top_n:].sum()
    if other > 0:
        out.loc['Other'] = other
    return out

py_phy = phylum_comp(py_best, top_n=7)
hz_phy = phylum_comp(hz_best, top_n=7)

phyla_union = list(pd.Index(py_phy.index).union(pd.Index(hz_phy.index)))
phyla_no_other = [p for p in phyla_union if str(p).lower() != 'other']
if 'Other' in phyla_union:
    phyla_union = phyla_no_other + ['Other']
else:
    phyla_union = phyla_no_other

phy_df = pd.DataFrame({
    'Puyang': py_phy.reindex(phyla_union).fillna(0).astype(int),
    'Hangzhou': hz_phy.reindex(phyla_union).fillna(0).astype(int)
})

# -----------------------------
# KO-profile PCA using top MAGs
# -----------------------------
sel_py = py_pa.loc[py_pa.index.intersection(py_top['MAG_ID'])].copy()
sel_hz = hz_pa.loc[hz_pa.index.intersection(hz_top['MAG_ID'])].copy()
all_kos = sel_py.columns.union(sel_hz.columns)
sel_py = sel_py.reindex(columns=all_kos, fill_value=0)
sel_hz = sel_hz.reindex(columns=all_kos, fill_value=0)
sel_pa = pd.concat([sel_py, sel_hz], axis=0)

ko_meta = pd.concat([
    py_top[['MAG_ID', 'variable', 'Phylum']].assign(site='Puyang'),
    hz_top[['MAG_ID', 'variable', 'Phylum']].assign(site='Hangzhou')
], ignore_index=True)

X = StandardScaler().fit_transform(sel_pa.values)
pca = PCA(n_components=2)
scores = pca.fit_transform(X)
expl = pca.explained_variance_ratio_ * 100
ko_meta['PC1'] = scores[:, 0]
ko_meta['PC2'] = scores[:, 1]

# -----------------------------
# Confidence ellipse
# -----------------------------
def add_confidence_ellipse(ax, x, y, facecolor, edgecolor, n_std=2.0, linewidth=0.8, alpha=0.18):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 3:
        return

    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.linalg.det(cov) == 0:
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle='--',
        linewidth=linewidth,
        alpha=alpha
    )
    ax.add_patch(ellipse)

# -----------------------------
# Plot Figure 2
# -----------------------------
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], width_ratios=[1.35, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# a: Puyang heatmap
im1 = ax1.imshow(
    py_heat_plot.values,
    aspect='auto',
    interpolation='nearest',
    vmin=heat_vmin,
#    cmap='magma',
    vmax=heat_vmax
)
ax1.set_yticks(np.arange(len(py_top)))
ax1.set_yticklabels(py_top['label'], fontsize=FS_TICK_Y)
ax1.set_xticks(np.arange(len(py_order)))
ax1.set_xticklabels(py_order, rotation=90, fontsize=FS_TICK_X)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.tick_params(axis='both', length=2.5, pad=1.5)
ax1.text(0.01, 1.02, 'a', transform=ax1.transAxes,
         fontsize=FS_PANEL, fontweight='bold', va='bottom')

# b: Hangzhou heatmap
im2 = ax2.imshow(
    hz_heat_plot.values,
    aspect='auto',
    interpolation='nearest',
    vmin=heat_vmin,
#    cmap='magma',
    vmax=heat_vmax
)
ax2.set_yticks(np.arange(len(hz_top)))
ax2.set_yticklabels(hz_top['label'], fontsize=FS_TICK_Y)
ax2.set_xticks(np.arange(len(hz_order)))
ax2.set_xticklabels(hz_order, rotation=90, fontsize=FS_TICK_X)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.tick_params(axis='both', length=2.5, pad=1.5)
ax2.text(0.01, 1.02, 'b', transform=ax2.transAxes,
         fontsize=FS_PANEL, fontweight='bold', va='bottom')

# One unified colorbar only at the right side of panel b
cbar = fig.colorbar(im2, ax=ax2, fraction=0.040, pad=0.02)
cbar.set_label('log10(abundance + 1e-6)', fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_TICK)

# c: phylum composition
x = np.arange(len(phy_df.index))
ax3.barh(
    x - 0.18, phy_df['Puyang'].values,
    height=0.35, label='Puyang', color=COLOR_PY,
    edgecolor='none', linewidth=0
)
ax3.barh(
    x + 0.18, phy_df['Hangzhou'].values,
    height=0.35, label='Hangzhou', color=COLOR_HZ,
    edgecolor='none', linewidth=0
)
ax3.set_yticks(x)
ax3.set_yticklabels(phy_df.index, fontsize=FS_TICK_Y)
ax3.invert_yaxis()
ax3.set_xlabel('Number of hydrocarbon-associated MAGs', fontsize=FS_AXIS)
ax3.tick_params(axis='x', labelsize=FS_TICK)
ax3.tick_params(axis='y', labelsize=FS_TICK_Y)
ax3.legend(frameon=False, fontsize=FS_LEGEND)
ax3.text(0.01, 1.02, 'c', transform=ax3.transAxes,
         fontsize=FS_PANEL, fontweight='bold', va='bottom')

# only keep left and bottom spines for panel c, and make them thinner
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(0.6)
ax3.spines['bottom'].set_linewidth(0.6)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')

# d: KO-profile PCA
for site_name, sub in ko_meta.groupby('site'):
    if site_name == 'Puyang':
        color = COLOR_PY
        marker = 'o'
    else:
        color = COLOR_HZ
        marker = 'o'

    add_confidence_ellipse(
        ax4, sub['PC1'], sub['PC2'],
        facecolor=color, edgecolor=color, n_std=2.0, linewidth=0.8, alpha=0.18
    )

    ax4.scatter(
        sub['PC1'], sub['PC2'],
        s=30, marker=marker, color=color, label=site_name,
        edgecolor='none', linewidth=0
    )

ax4.axhline(0, linewidth=0.6, color='#999999')
ax4.axvline(0, linewidth=0.6, color='#999999')
ax4.set_xlabel(f'PC1 ({expl[0]:.1f}%)', fontsize=FS_AXIS)
ax4.set_ylabel(f'PC2 ({expl[1]:.1f}%)', fontsize=FS_AXIS)
ax4.tick_params(axis='both', labelsize=FS_TICK)
ax4.legend(frameon=False, fontsize=FS_LEGEND)
ax4.text(0.01, 1.02, 'd', transform=ax4.transAxes,
         fontsize=FS_PANEL, fontweight='bold', va='bottom')

# only keep left and bottom spines for panel d, and make them thinner
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_linewidth(0.6)
ax4.spines['bottom'].set_linewidth(0.6)
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')

out_png = Path('Figure2.png')
out_pdf = Path('Figure2.pdf')
out_tif = Path('Figure2.tif')

fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_tif, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Saved PNG: {out_png}")
print(f"Saved PDF: {out_pdf}")
print(f"Saved TIFF: {out_tif}")