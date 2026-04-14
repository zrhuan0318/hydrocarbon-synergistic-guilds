import os
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


# Global style for main-text figure
# -----------------------------
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
})

# Main-text figure size / resolution
FIG_W = 7.2
FIG_H = 6.6
DPI = 600

# Unified font sizes
FS_TICK = 6.6
FS_LABEL = 7.8
FS_HEAT_X = 6.2
FS_HEAT_Y = 6.0
FS_CBAR = 6.8
FS_LEGEND = 6.2
FS_PANEL = 9.0


# Load data
# -----------------------------
sample = pd.read_csv('1_Sample_metadata_and_inclusion_status.txt', sep='\t')
py_ab = pd.read_csv('3_MAGs_abundance_Puyang.txt', sep='\t')
hz_ab = pd.read_csv('4_MAGs_abundance_Hangzhou.txt', sep='\t')
py_ann = pd.read_csv('5_MAGs_taxonomic_classification_Puyang.txt', sep='\t').rename(columns={'MAG ID': 'MAG_ID'})
hz_ann = pd.read_csv('5_MAGs_taxonomic_classification_Hangzhou.txt', sep='\t')
py_fun = pd.read_csv('6_MAGs_function_annotation_Puyang.txt', sep='\t', low_memory=False)
hz_fun = pd.read_csv('6_MAGs_function_annotation_Hangzhou.txt', sep='\t', low_memory=False)
py_env = pd.read_csv('2_Environmental_factors_and_pollutant_profiles_Puyang.txt', sep='\t')
hz_env = pd.read_csv('2_Environmental_factors_and_pollutant_profiles_Hangzhou.txt', sep='\t')


# Sample metadata
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


# Abundance matrices
def prep_abundance(ab):
    ab = ab.copy()
    ab = ab.rename(columns={ab.columns[0]: 'MAG_ID'})
    return ab

py_ab2 = prep_abundance(py_ab)
hz_ab2 = prep_abundance(hz_ab)

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


# KO matrix and curated modules
def ko_matrix(fun_df):
    tmp = fun_df.copy().rename(columns={fun_df.columns[0]: 'KO'})
    for col in tmp.columns[1:]:
        tmp[col] = pd.to_numeric(tmp[col], errors='coerce').fillna(0)
    return tmp.set_index('KO')

py_ko = ko_matrix(py_fun)
hz_ko = ko_matrix(hz_fun)

modules = {
    'Hydrocarbon activation / oxygenation': [
        'K00496', 'K20938', 'K00480', 'K16246', 'K14579'
    ],
    'Aromatic-ring cleavage / central aromatic metabolism': [
        'K03381', 'K03382', 'K03383', 'K03384',
        'K07104', 'K07105', 'K15253', 'K15254',
        'K00446', 'K01856'
    ],
    'Beta-oxidation / fatty-acid degradation': [
        'K07511', 'K07512', 'K01073', 'K01782', 'K00626', 'K00074'
    ],
    'Fermentation / syntrophic intermediate metabolism': [
        'K00169', 'K00174', 'K03737', 'K03738',
        'K00156', 'K00157', 'K00158', 'K00625', 'K13923'
    ],
    'Methanogenesis': [
        'K00399', 'K00401', 'K00402', 'K00400',
        'K14080', 'K14082', 'K14084',
        'K11260', 'K11261', 'K11262'
    ],
    'Sulfate reduction / sulfur respiration': [
        'K00958', 'K00394', 'K00395', 'K11180',
        'K11181'
    ],
    'Nitrate reduction / denitrification': [
        'K00370', 'K00371', 'K02567', 'K02568',
        'K00368', 'K15864', 'K04561'
    ]
}

module_abbrev = {
    'Hydrocarbon activation / oxygenation': 'HA/O',
    'Aromatic-ring cleavage / central aromatic metabolism': 'AC/CAM',
    'Beta-oxidation / fatty-acid degradation': 'B/FD',
    'Fermentation / syntrophic intermediate metabolism': 'F/SIM',
    'Methanogenesis': 'MG',
    'Sulfate reduction / sulfur respiration': 'SR/SR',
    'Nitrate reduction / denitrification': 'NR/D'
}

def module_scores(ko_mat, mag_ids):
    mag_ids = [m for m in mag_ids if m in ko_mat.columns]
    out = pd.DataFrame(index=mag_ids)
    for mod, kos in modules.items():
        present = [k for k in kos if k in ko_mat.index]
        if len(present) == 0:
            out[mod] = 0.0
        else:
            out[mod] = (ko_mat.loc[present, mag_ids] > 0).sum(axis=0) / len(present)
    return out

py_top = py_best.head(20).copy()
hz_top = hz_best.head(20).copy()

def add_labels(top_df):
    top_df = top_df.copy()
    top_df['label'] = top_df.apply(
        lambda r: f"{r['MAG_ID']} | {r['Phylum']}" if pd.isna(r['Genus']) else f"{r['MAG_ID']} | {r['Genus']}",
        axis=1
    )
    return top_df

py_top = add_labels(py_top)
hz_top = add_labels(hz_top)

py_mod = module_scores(py_ko, py_top['MAG_ID'].tolist())
hz_mod = module_scores(hz_ko, hz_top['MAG_ID'].tolist())

py_mod.index = py_top.set_index('MAG_ID').loc[py_mod.index, 'label']
hz_mod.index = hz_top.set_index('MAG_ID').loc[hz_mod.index, 'label']

py_mod = py_mod.rename(columns=module_abbrev)
hz_mod = hz_mod.rename(columns=module_abbrev)

# Summaries by site and by main associated variable
def summarize_modules(mod_df, top_df):
    top_df = top_df.set_index('label')
    joined = mod_df.copy()
    joined['main_assoc'] = top_df.loc[joined.index, 'variable']
    site_mean = joined.drop(columns=['main_assoc']).mean()
    by_assoc = joined.groupby('main_assoc').mean(numeric_only=True)
    return site_mean, by_assoc

py_site_mean, py_by_assoc = summarize_modules(py_mod, py_top)
hz_site_mean, hz_by_assoc = summarize_modules(hz_mod, hz_top)

module_summary = pd.DataFrame({
    'Puyang_top20_mean': py_site_mean,
    'Hangzhou_top20_mean': hz_site_mean
})


# Plot Figure 3
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

# 外层：上下两排
outer = fig.add_gridspec(
    2, 1,
    height_ratios=[1.28, 1.0],
    hspace=0.30
)

# 上排：保持 a/b 很大的间距
top_gs = outer[0].subgridspec(
    1, 2,
    width_ratios=[1.00, 1.00],
    wspace=0.95
)

# 下排：让 c/d 更宽，间距更正常
bottom_gs = outer[1].subgridspec(
    1, 2,
    width_ratios=[1.05, 1.05],
    wspace=0.28
)

ax1 = fig.add_subplot(top_gs[0, 0])
ax2 = fig.add_subplot(top_gs[0, 1])
ax3 = fig.add_subplot(bottom_gs[0, 0])
ax4 = fig.add_subplot(bottom_gs[0, 1])

# a: Puyang module heatmap
vmax = max(py_mod.max().max(), hz_mod.max().max(), 1e-6)
im1 = ax1.imshow(
    py_mod.values,
    aspect='auto',
    interpolation='nearest',
    vmin=0,
#    cmap='magma',
    vmax=vmax
)
ax1.set_yticks(np.arange(py_mod.shape[0]))
ax1.set_yticklabels(py_mod.index, fontsize=FS_HEAT_Y)
ax1.set_xticks(np.arange(py_mod.shape[1]))
ax1.set_xticklabels(py_mod.columns, rotation=45, ha='right', fontsize=FS_HEAT_X)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.032, pad=0.02)
cbar1.set_label('Module score', fontsize=FS_CBAR)
cbar1.ax.tick_params(labelsize=FS_TICK)
ax1.text(0.01, 1.02, 'a', transform=ax1.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')

# b: Hangzhou module heatmap
im2 = ax2.imshow(
    hz_mod.values,
    aspect='auto',
    interpolation='nearest',
    vmin=0,
#    cmap='magma',
    vmax=vmax
)
ax2.set_yticks(np.arange(hz_mod.shape[0]))
ax2.set_yticklabels(hz_mod.index, fontsize=FS_HEAT_Y)
ax2.set_xticks(np.arange(hz_mod.shape[1]))
ax2.set_xticklabels(hz_mod.columns, rotation=45, ha='right', fontsize=FS_HEAT_X)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.032, pad=0.02)
cbar2.set_label('Module score', fontsize=FS_CBAR)
cbar2.ax.tick_params(labelsize=FS_TICK)
ax2.text(0.01, 1.02, 'b', transform=ax2.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')

# c: site-level module comparison
mods = module_summary.index.tolist()
x = np.arange(len(mods))
width = 0.38
ax3.barh(
    x - width/2,
    module_summary['Puyang_top20_mean'].values,
    height=width,
    label='Puyang',
    color='#E69F00',
    edgecolor='none',
    linewidth=0
)

ax3.barh(
    x + width/2,
    module_summary['Hangzhou_top20_mean'].values,
    height=width,
    label='Hangzhou',
    color='#56B4E9',
    edgecolor='none',
    linewidth=0
)

ax3.set_yticks(x)
ax3.set_yticklabels(mods, fontsize=FS_TICK)
ax3.invert_yaxis()
ax3.set_xlabel('Mean module score across top 20 MAGs', fontsize=FS_LABEL)
ax3.legend(frameon=False, fontsize=FS_LEGEND)
ax3.text(0.01, 1.02, 'c', transform=ax3.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')

# c: only left and bottom axes, thinner spines
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(0.6)
ax3.spines['bottom'].set_linewidth(0.6)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')

# d: by associated fraction
comb = []
for site_name, by_assoc in [('Puyang', py_by_assoc), ('Hangzhou', hz_by_assoc)]:
    tmp = by_assoc.copy()
    tmp['site'] = site_name
    tmp['main_assoc'] = tmp.index
    comb.append(tmp.reset_index(drop=True))
comb = pd.concat(comb, ignore_index=True)

group_labels = []
vals = []
for _, row in comb.iterrows():
    group_labels.append(f"{row['site']} | {row['main_assoc']}")
    vals.append([row[m] for m in mods])
vals = np.array(vals)

y = np.arange(len(mods))
d_colors = ['#81DACE', '#FF9B99', '#9BC9FF', '#9B9ACA']
offsets = np.linspace(-0.3, 0.3, vals.shape[0]) if vals.shape[0] > 1 else [0]
for i, label in enumerate(group_labels):
    ax4.barh(
        y + offsets[i],
        vals[i],
        height=0.23,
        label=label,
        color=d_colors[i],
        edgecolor='none',
        linewidth=0
    )

ax4.set_yticks(y)
ax4.set_yticklabels(mods, fontsize=FS_TICK)
ax4.invert_yaxis()
ax4.set_xlabel('Mean module score', fontsize=FS_LABEL)
ax4.legend(frameon=False, fontsize=FS_LEGEND, loc='lower right')
ax4.text(0.01, 1.02, 'd', transform=ax4.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom')

# d: only left and bottom axes, thinner spines
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_linewidth(0.6)
ax4.spines['bottom'].set_linewidth(0.6)
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(labelsize=FS_TICK)

fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.10)

out_png = Path('Figure3_600dpi.png')
out_pdf = Path('Figure3_600dpi.pdf')
out_tif = Path('Figure3_600dpi.tif')

fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(out_tif, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Saved PNG: {out_png}")
print(f"Saved PDF: {out_pdf}")
print(f"Saved TIFF: {out_tif}")
