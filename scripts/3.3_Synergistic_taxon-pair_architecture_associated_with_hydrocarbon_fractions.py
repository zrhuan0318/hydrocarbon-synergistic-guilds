import os
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# -----------------------------
# Global style
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
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

DPI = 600

# Keep panel-like sizes roughly similar, but enlarge network panels a bit
SIZE_NET = (2.55, 2.25)   # a-d enlarged to avoid label clipping
SIZE_E = (2.70, 2.00)     # e about 10% wider
SIZE_F = (2.20, 2.00)     # f about 10% narrower

FS_AXIS = 7.6
FS_TICK = 6.4
FS_LABEL = 6.0
FS_LEGEND = 6.0

# -----------------------------
# Input files
# -----------------------------
files = {
    'Puyang C10-C40': 'Puyang_C10C40_sid_ready_sid_pairwise_network_annotated.tsv',
    'Hangzhou C10-C40': 'Hangzhou_C10C40_sid_ready_sid_pairwise_network_annotated.tsv',
    'Puyang C6-C9': 'Puyang_C6C9_sid_ready_sid_pairwise_network_annotated.tsv',
    'Hangzhou C6-C9': 'Hangzhou_C6C9_sid_ready_sid_pairwise_network_annotated.tsv',
}

out_dir = Path('')
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def short_tax(genus, phylum):
    if pd.notna(genus) and str(genus).strip() != "":
        return str(genus).strip()
    if pd.notna(phylum) and str(phylum).strip() != "":
        return str(phylum).strip()
    return "Unclassified"

def normalize_phylum(p):
    if pd.isna(p) or str(p).strip() == "":
        return "Unclassified"
    return str(p).strip()

def load_network_table(path):
    df = pd.read_csv(path, sep='\t')
    required = ['synergy', 'source_genus', 'source_phylum', 'target_genus', 'target_phylum']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df['source_tax'] = [short_tax(g, p) for g, p in zip(df['source_genus'], df['source_phylum'])]
    df['target_tax'] = [short_tax(g, p) for g, p in zip(df['target_genus'], df['target_phylum'])]
    df['source_phy'] = [normalize_phylum(p) for p in df['source_phylum']]
    df['target_phy'] = [normalize_phylum(p) for p in df['target_phylum']]

    df = df[df['source_tax'] != df['target_tax']].copy()

    pair_a = np.minimum(df['source_tax'].astype(str), df['target_tax'].astype(str))
    pair_b = np.maximum(df['source_tax'].astype(str), df['target_tax'].astype(str))
    df['pair_key'] = pair_a + ' || ' + pair_b

    df = df.sort_values('synergy', ascending=False).drop_duplicates('pair_key').copy()
    return df

def top_edges(df, n=16):
    return df.sort_values('synergy', ascending=False).head(n).copy()

def build_graph(df_top):
    G = nx.Graph()
    for _, r in df_top.iterrows():
        s = r['source_tax']
        t = r['target_tax']
        sy = float(r['synergy'])
        sphy = r['source_phy']
        tphy = r['target_phy']

        if not G.has_node(s):
            G.add_node(s, phylum=sphy)
        if not G.has_node(t):
            G.add_node(t, phylum=tphy)

        G.add_edge(s, t, synergy=sy)
    return G

def node_metrics_from_top(df_top):
    rows = []
    for _, r in df_top.iterrows():
        rows.append((r['source_tax'], r['source_phy'], r['synergy']))
        rows.append((r['target_tax'], r['target_phy'], r['synergy']))
    tmp = pd.DataFrame(rows, columns=['taxon', 'phylum', 'synergy'])
    freq = tmp.groupby('taxon').size().rename('freq')
    sy_sum = tmp.groupby('taxon')['synergy'].sum().rename('sy_sum')
    phylum = tmp.groupby('taxon')['phylum'].first().rename('phylum')
    out = pd.concat([freq, sy_sum, phylum], axis=1).reset_index().rename(columns={'index': 'taxon'})
    return out

def make_phylum_palette(all_phyla):
    base = [
        '#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
        '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC', '#8E6C8A',
        '#2E86AB', '#C73E1D'
    ]
    phyla_sorted = sorted(all_phyla)
    pal = {}
    for i, p in enumerate(phyla_sorted):
        pal[p] = base[i % len(base)]
    pal['Unclassified'] = '#BDBDBD'
    return pal

def expand_axis_limits_from_pos(ax, pos, pad=0.18):
    xs = np.array([xy[0] for xy in pos.values()])
    ys = np.array([xy[1] for xy in pos.values()])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0

    ax.set_xlim(xmin - pad * xspan, xmax + pad * xspan)
    ax.set_ylim(ymin - pad * yspan, ymax + pad * yspan)

def plot_network_panel(ax, df_top, phylum_colors, seed=42, max_labels=10):
    G = build_graph(df_top)
    node_df = node_metrics_from_top(df_top).copy()

    pos = nx.spring_layout(
        G,
        seed=seed,
        k=1.0 / max(np.sqrt(len(G.nodes)), 1),
        iterations=300
    )

    node_df['label_score'] = node_df['freq'] * 2 + node_df['sy_sum']
    label_nodes = set(
        node_df.sort_values(['label_score', 'freq', 'sy_sum'], ascending=False)
        .head(max_labels)['taxon']
        .tolist()
    )

    node_sizes = []
    node_colors = []
    labels = {}
    for n in G.nodes():
        row = node_df[node_df['taxon'] == n].iloc[0]
        node_sizes.append(40 + row['freq'] * 26 + row['sy_sum'] * 180)
        node_colors.append(phylum_colors.get(row['phylum'], '#BDBDBD'))
        labels[n] = n if n in label_nodes else ""

    edge_sy = np.array([G[u][v]['synergy'] for u, v in G.edges()])
    if len(edge_sy) == 1:
        widths = [1.8]
    else:
        widths = 0.8 + 2.8 * (edge_sy - edge_sy.min()) / (edge_sy.max() - edge_sy.min() + 1e-12)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=widths,
        edge_color='#B0B0B0',
        alpha=0.75
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=0.7
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=labels,
        font_size=FS_LABEL
    )

    # Critical fix: add margin around outermost nodes so labels don't get clipped
    expand_axis_limits_from_pos(ax, pos, pad=0.20)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_visible(False)

def recurrence_table(top_dict):
    rows = []
    for analysis, df_top in top_dict.items():
        for _, r in df_top.iterrows():
            rows.append((analysis, r['source_tax'], r['source_phy'], r['synergy']))
            rows.append((analysis, r['target_tax'], r['target_phy'], r['synergy']))
    tmp = pd.DataFrame(rows, columns=['analysis', 'taxon', 'phylum', 'synergy'])

    out = tmp.groupby(['analysis', 'taxon', 'phylum']).agg(
        freq=('taxon', 'size'),
        max_synergy=('synergy', 'max'),
        sum_synergy=('synergy', 'sum')
    ).reset_index()

    taxa_keep = (
        out.groupby('taxon')
        .agg(n_analysis=('analysis', 'nunique'),
             total_freq=('freq', 'sum'),
             top_synergy=('max_synergy', 'max'))
        .sort_values(['n_analysis', 'total_freq', 'top_synergy'], ascending=[False, False, False])
    )

    taxa_keep = taxa_keep.head(12).index.tolist()
    out = out[out['taxon'].isin(taxa_keep)].copy()

    order = (
        out.groupby('taxon')
        .agg(n_analysis=('analysis', 'nunique'),
             total_freq=('freq', 'sum'),
             top_synergy=('max_synergy', 'max'))
        .sort_values(['n_analysis', 'total_freq', 'top_synergy'], ascending=[False, False, False])
        .index.tolist()
    )
    return out, order

def save_figure(fig, stem):
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    tif = out_dir / f"{stem}.tif"
    fig.savefig(png, dpi=DPI, bbox_inches='tight', transparent=True)
    fig.savefig(pdf, dpi=DPI, bbox_inches='tight', transparent=True)
    fig.savefig(tif, dpi=DPI, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {tif}")

# -----------------------------
# Load all data
# -----------------------------
dfs = {}
for key, path in files.items():
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    dfs[key] = load_network_table(path)

top_dict = {k: top_edges(v, n=16) for k, v in dfs.items()}

all_phyla = []
for df_top in top_dict.values():
    all_phyla.extend(df_top['source_phy'].tolist())
    all_phyla.extend(df_top['target_phy'].tolist())
phylum_colors = make_phylum_palette(sorted(set(all_phyla)))

rank_df = []
for analysis, df in dfs.items():
    d = df.sort_values('synergy', ascending=False).reset_index(drop=True).copy()
    d['rank'] = np.arange(1, len(d) + 1)
    d['analysis'] = analysis
    rank_df.append(d[['analysis', 'rank', 'synergy']])
rank_df = pd.concat(rank_df, ignore_index=True)

rec_df, taxa_order = recurrence_table(top_dict)

analysis_order = ['Puyang C10-C40', 'Hangzhou C10-C40', 'Puyang C6-C9', 'Hangzhou C6-C9']
xlabels_short = ['PY C10–C40', 'HZ C10–C40', 'PY C6–C9', 'HZ C6–C9']

site_colors = {
    'Puyang': '#4C78A8',
    'Hangzhou': '#E45756'
}
fraction_styles = {
    'C10-C40': '-',
    'C6-C9': '--'
}

# -----------------------------
# a-d separate network panels
# -----------------------------
network_specs = [
    ('Figure4_a_network_PY_C10C40', 'Puyang C10-C40', 11),
    ('Figure4_b_network_HZ_C10C40', 'Hangzhou C10-C40', 13),
    ('Figure4_c_network_PY_C6C9', 'Puyang C6-C9', 17),
    ('Figure4_d_network_HZ_C6C9', 'Hangzhou C6-C9', 19),
]

for stem, key, seed in network_specs:
    fig, ax = plt.subplots(figsize=SIZE_NET, dpi=DPI)
    plot_network_panel(ax, top_dict[key], phylum_colors, seed=seed, max_labels=10)
    save_figure(fig, stem)

# -----------------------------
# e separate rank-synergy panel
# -----------------------------
fig, ax = plt.subplots(figsize=SIZE_E, dpi=DPI)

for analysis in analysis_order:
    sub = rank_df[rank_df['analysis'] == analysis].copy()
    site = 'Puyang' if analysis.startswith('Puyang') else 'Hangzhou'
    fraction = 'C10-C40' if 'C10-C40' in analysis else 'C6-C9'
    ax.plot(
        sub['rank'], sub['synergy'],
        lw=1.7,
        color=site_colors[site],
        linestyle=fraction_styles[fraction]
    )

ax.set_xlabel('Rank of synergistic taxon pairs', fontsize=FS_AXIS)
ax.set_ylabel('Pairwise synergy value', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)

handles = [
    mpl.lines.Line2D([0], [0], color=site_colors['Puyang'], lw=1.7, linestyle='-', label='PY C10–C40'),
    mpl.lines.Line2D([0], [0], color=site_colors['Hangzhou'], lw=1.7, linestyle='-', label='HZ C10–C40'),
    mpl.lines.Line2D([0], [0], color=site_colors['Puyang'], lw=1.7, linestyle='--', label='PY C6–C9'),
    mpl.lines.Line2D([0], [0], color=site_colors['Hangzhou'], lw=1.7, linestyle='--', label='HZ C6–C9'),
]
ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_figure(fig, 'Figure4_e_rank_synergy')

# -----------------------------
# f separate taxa recurrence panel
# -----------------------------
fig, ax = plt.subplots(figsize=SIZE_F, dpi=DPI)

x_map = {a: i for i, a in enumerate(analysis_order)}
y_map = {t: i for i, t in enumerate(taxa_order)}

for _, r in rec_df.iterrows():
    ax.scatter(
        x_map[r['analysis']],
        y_map[r['taxon']],
        s=18 + r['freq'] * 26,
        c=[phylum_colors.get(r['phylum'], '#BDBDBD')],
        alpha=0.9,
        edgecolors='white',
        linewidths=0.5
    )

ax.set_xticks(range(len(analysis_order)))
ax.set_xticklabels(xlabels_short, rotation=25, ha='right', fontsize=FS_TICK)
ax.set_yticks(range(len(taxa_order)))
ax.set_yticklabels(taxa_order, fontsize=FS_TICK)
ax.invert_yaxis()
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

phyla_in_top = []
for tax in taxa_order:
    p = rec_df.loc[rec_df['taxon'] == tax, 'phylum'].iloc[0]
    if p not in phyla_in_top:
        phyla_in_top.append(p)

handles_phy = [
    mpl.lines.Line2D([0], [0], marker='o', color='none',
                     markerfacecolor=phylum_colors[p], markeredgecolor='white',
                     markeredgewidth=0.5, markersize=5.2, label=p)
    for p in phyla_in_top[:8]
]
ax.legend(
    handles=handles_phy,
    frameon=False,
    fontsize=FS_LEGEND,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    title='Phylum',
    title_fontsize=FS_LEGEND,
    borderaxespad=0.0
)

save_figure(fig, 'Figure4_f_taxa_recurrence')