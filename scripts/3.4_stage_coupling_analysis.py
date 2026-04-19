#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3.4 analysis script
- dominant stage assignment
- stage-coupling matrix
- stage-coupling composition
- stage-coupling index (SCI)
- depth-resolved module/stage succession

This script only performs analysis and exports tables.
Plotting should be done in a separate script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd



# 1. Module definitions
# =========================
MODULES = {
    'HA/O': ['K00496', 'K20938', 'K00480', 'K16246', 'K14579'],
    'AC/CAM': ['K03381', 'K03382', 'K03383', 'K03384', 'K07104', 'K07105', 'K15253', 'K15254', 'K00446', 'K01856'],
    'B/FD': ['K07511', 'K07512', 'K01073', 'K01782', 'K00626', 'K00074'],
    'F/SIM': ['K00169', 'K00174', 'K03737', 'K03738', 'K00156', 'K00157', 'K00158', 'K00625', 'K13923'],
    'MG': ['K00399', 'K00401', 'K00402', 'K00400', 'K14080', 'K14082', 'K14084', 'K11260', 'K11261', 'K11262'],
    'SR/SR': ['K00958', 'K00394', 'K00395', 'K11180', 'K11181'],
    'NR/D': ['K00370', 'K00371', 'K02567', 'K02568', 'K00368', 'K15864', 'K04561'],
}

STAGE_MAP = {
    'HA/O': 'Stage I',
    'AC/CAM': 'Stage I',
    'B/FD': 'Stage II',
    'F/SIM': 'Stage II',
    'NR/D': 'Stage III',
    'SR/SR': 'Stage III',
    'MG': 'Stage III',
}

STAGE_ORDER = ['Stage I', 'Stage II', 'Stage III']



# 2. Helpers
# =========================
def read_ko_matrix(fun_file: str) -> pd.DataFrame:
    df = pd.read_csv(fun_file, sep='\t', low_memory=False)
    df = df.rename(columns={df.columns[0]: 'KO'})
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df.set_index('KO')


def module_scores_from_ko(ko_mat: pd.DataFrame, mag_ids: list[str]) -> pd.DataFrame:
    mag_ids = [m for m in mag_ids if m in ko_mat.columns]
    out = pd.DataFrame(index=mag_ids)

    for mod, kos in MODULES.items():
        present = [k for k in kos if k in ko_mat.index]
        if len(present) == 0:
            out[mod] = 0.0
        else:
            out[mod] = (ko_mat.loc[present, mag_ids] > 0).sum(axis=0) / len(present)

    return out


def stage_scores_from_module_scores(mod_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=mod_df.index)
    for stage in STAGE_ORDER:
        cols = [m for m, s in STAGE_MAP.items() if s == stage]
        out[stage] = mod_df[cols].sum(axis=1)
    return out


def dominant_stage(stage_df: pd.DataFrame) -> pd.Series:
    return stage_df.idxmax(axis=1)


def stage_distance(stage_a: str, stage_b: str) -> int:
    idx = {s: i for i, s in enumerate(STAGE_ORDER)}
    return abs(idx[stage_a] - idx[stage_b])


def load_abundance_matrix(abundance_file: str) -> pd.DataFrame:
    df = pd.read_csv(abundance_file, sep='\t')
    df = df.rename(columns={df.columns[0]: 'MAG_ID'})
    return df.set_index('MAG_ID')


def load_depth_info(sample_file: str, site_name: str) -> pd.DataFrame:
    s = pd.read_csv(sample_file, sep='\t')
    s = s[s['是否参与 MAG abundance 分析'].astype(str).str.lower() == 'yes'].copy()
    s['site_norm'] = s['site'].astype(str).str.strip().str.lower()
    s['depth'] = pd.to_numeric(s['depth'], errors='coerce')

    site_norm = site_name.strip().lower()
    s = s[s['site_norm'] == site_norm][['sample_ID', 'depth']].copy()

    # user-confirmed fix for Puyang
    if site_norm == 'puyang':
        s['sample_ID'] = s['sample_ID'].replace({'SB2-2': 'SB2-1'})
        s = s.drop_duplicates(subset=['sample_ID'])

    return s


def assign_depth_bin(depth: float) -> str:
    if pd.isna(depth):
        return 'Unknown'
    if depth < 1.5:
        return '0-1.5 m'
    elif depth < 3.0:
        return '1.5-3.0 m'
    else:
        return '3.0-5.0 m'



# 3. Main analysis
# =========================
def main():
    parser = argparse.ArgumentParser(description='3.4 stage-coupling analysis')
    parser.add_argument('--site', required=True, choices=['Puyang', 'Hangzhou'])
    parser.add_argument('--pairwise_network', required=True,
                        help='Annotated SID pairwise network table')
    parser.add_argument('--ko_annotation', required=True,
                        help='KO annotation table')
    parser.add_argument('--abundance', required=True,
                        help='MAG abundance table')
    parser.add_argument('--sample_info', required=True,
                        help='Sample metadata table')
    parser.add_argument('--topn_pairs', type=int, default=10,
                        help='Number of top synergy pairs to analyze')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- read inputs ----------
    pair_df = pd.read_csv(args.pairwise_network, sep='\t')
    pair_df = pair_df.sort_values('synergy', ascending=False).head(args.topn_pairs).copy()

    ko_mat = read_ko_matrix(args.ko_annotation)
    ab = load_abundance_matrix(args.abundance)
    depth_df = load_depth_info(args.sample_info, args.site)

    # ---------- collect MAGs in top pairs ----------
    top_mags = pd.unique(pair_df[['source_otu', 'target_otu']].values.ravel()).tolist()

    # ---------- module scores ----------
    mod_df = module_scores_from_ko(ko_mat, top_mags)
    mod_df.index.name = 'MAG_ID'
    mod_df.to_csv(outdir / f'{args.site}_topMAG_module_scores.csv')

    # ---------- stage scores ----------
    stage_df = stage_scores_from_module_scores(mod_df)
    stage_df.index.name = 'MAG_ID'
    stage_df.to_csv(outdir / f'{args.site}_topMAG_stage_scores.csv')

    dom_stage = dominant_stage(stage_df).rename('dominant_stage')
    dom_stage.to_csv(outdir / f'{args.site}_topMAG_dominant_stage.csv', header=True)

    # ---------- pair-level stage assignment ----------
    pair_rows = []
    for _, r in pair_df.iterrows():
        s = r['source_otu']
        t = r['target_otu']
        if s not in stage_df.index or t not in stage_df.index:
            continue

        s_stage = dom_stage.loc[s]
        t_stage = dom_stage.loc[t]
        dist = stage_distance(s_stage, t_stage)

        pair_rows.append({
            'source_otu': s,
            'target_otu': t,
            'synergy': r['synergy'],
            'redundant': r.get('redundant', np.nan),
            'source_stage': s_stage,
            'target_stage': t_stage,
            'stage_distance': dist,
            'source_tax': r.get('source_genus', np.nan) if pd.notna(r.get('source_genus', np.nan)) else r.get('source_phylum', np.nan),
            'target_tax': r.get('target_genus', np.nan) if pd.notna(r.get('target_genus', np.nan)) else r.get('target_phylum', np.nan),
        })

    pair_stage_df = pd.DataFrame(pair_rows)
    pair_stage_df.to_csv(outdir / f'{args.site}_pair_stage_assignment.csv', index=False)

    # ---------- stage-coupling matrix (count) ----------
    count_mat = pd.pivot_table(
        pair_stage_df,
        index='source_stage',
        columns='target_stage',
        values='synergy',
        aggfunc='count',
        fill_value=0
    ).reindex(index=STAGE_ORDER, columns=STAGE_ORDER, fill_value=0)
    count_mat.to_csv(outdir / f'{args.site}_stage_coupling_count_matrix.csv')

    # ---------- stage-coupling matrix (synergy-weighted) ----------
    weight_mat = pd.pivot_table(
        pair_stage_df,
        index='source_stage',
        columns='target_stage',
        values='synergy',
        aggfunc='sum',
        fill_value=0
    ).reindex(index=STAGE_ORDER, columns=STAGE_ORDER, fill_value=0)
    weight_mat.to_csv(outdir / f'{args.site}_stage_coupling_weighted_matrix.csv')

    # ---------- stage-coupling composition ----------
    comp = (
        pair_stage_df
        .assign(stage_pair=lambda d: d['source_stage'] + ' -> ' + d['target_stage'])
        .groupby('stage_pair', as_index=False)
        .agg(
            n_pairs=('synergy', 'size'),
            synergy_sum=('synergy', 'sum'),
            synergy_mean=('synergy', 'mean')
        )
        .sort_values('synergy_sum', ascending=False)
    )
    total_synergy = comp['synergy_sum'].sum()
    comp['synergy_weighted_proportion'] = comp['synergy_sum'] / total_synergy if total_synergy > 0 else 0
    comp.to_csv(outdir / f'{args.site}_stage_coupling_composition.csv', index=False)

    # ---------- stage-coupling index ----------
    sci_num = (pair_stage_df['synergy'] * pair_stage_df['stage_distance']).sum()
    sci_den = pair_stage_df['synergy'].sum()
    sci = sci_num / sci_den if sci_den > 0 else np.nan
    sci_df = pd.DataFrame([{
        'site': args.site,
        'topn_pairs': args.topn_pairs,
        'stage_coupling_index': sci
    }])
    sci_df.to_csv(outdir / f'{args.site}_stage_coupling_index.csv', index=False)

    # ---------- depth-resolved succession ----------
    # mean abundance per MAG within each depth bin
    common_samples = [c for c in ab.columns if c in depth_df['sample_ID'].tolist()]
    depth_df = depth_df[depth_df['sample_ID'].isin(common_samples)].copy()
    depth_df['depth_bin'] = depth_df['depth'].apply(assign_depth_bin)

    # keep only top synergy MAGs
    ab_top = ab.loc[[m for m in top_mags if m in ab.index], common_samples].copy()

    depth_rows = []
    for d_bin, sub in depth_df.groupby('depth_bin'):
        samples = sub['sample_ID'].tolist()
        if len(samples) == 0:
            continue

        mean_ab = ab_top[samples].mean(axis=1)

        # abundance-weighted module score
        weighted_mod = (mod_df.loc[mean_ab.index].multiply(mean_ab, axis=0)).sum(axis=0)
        if mean_ab.sum() > 0:
            weighted_mod = weighted_mod / mean_ab.sum()

        weighted_stage = {
            stage: weighted_mod[[m for m, s in STAGE_MAP.items() if s == stage]].sum()
            for stage in STAGE_ORDER
        }

        row = {'depth_bin': d_bin}
        for m in mod_df.columns:
            row[m] = weighted_mod[m]
        for s in STAGE_ORDER:
            row[s] = weighted_stage[s]
        depth_rows.append(row)

    depth_succession_df = pd.DataFrame(depth_rows)
    depth_order = ['0-1.5 m', '1.5-3.0 m', '3.0-5.0 m']
    if not depth_succession_df.empty:
        depth_succession_df['depth_bin'] = pd.Categorical(depth_succession_df['depth_bin'], categories=depth_order, ordered=True)
        depth_succession_df = depth_succession_df.sort_values('depth_bin')

    depth_succession_df.to_csv(outdir / f'{args.site}_depth_resolved_module_succession.csv', index=False)

    # ---------- functional relay path hint ----------
    # rank dominant stage-pair paths by weighted synergy
    relay_df = comp[['stage_pair', 'synergy_sum', 'synergy_weighted_proportion']].copy()
    relay_df.to_csv(outdir / f'{args.site}_functional_relay_paths.csv', index=False)

    print(f'[DONE] 3.4 analysis finished for {args.site}')
    print(f'Output dir: {outdir}')


if __name__ == '__main__':
    main()
