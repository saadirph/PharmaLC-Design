import pandas as pd, numpy as np, json
D = '/home/claude/data/'
N = json.load(open(D + 'numbers.json')); M = N['meta']; SY = N['sys']
T = {}
S = pd.read_csv(D + 'kb2_systems.csv', dtype={'dataset': str}); P = pd.read_csv(D + 'kb2_predictions.csv', dtype={'dataset': str})
T['t1'] = [['Retention records', f"{M['rows']:,}"], ['Unique compounds (InChIKey connectivity)', f"{M['compounds']:,}"], ['Chromatographic systems (gradient RP)', str(M['datasets'])],
           ['Systems with a linked publication / distinct DOIs or URLs', f"{int(S.url.notna().sum())} / {SY['studies']}"], ['Distinct columns', str(SY['cols'])], ['USP L1 (C18) / L11 (phenyl) systems', f"{SY['L1']} / {SY['L11']}"],
           ['Systems with HSM / Tanaka parameters', f"{SY['hsm']} / {SY['tanaka']}"], ['Acetonitrile-based systems', str(SY['acn'])], ['Systems with formic acid', str(SY['formic'])],
           ['Median flow rate (mL min^{−1}) / t_{0} (min)', f"{SY['flow_med']:.2f} / {SY['t0_med']:.2f}"], ['Molecular / system descriptors', f"{M['n_mol']} / {M['n_sys']}"],
           ['Drug panel: drugs / records', f"{M['drug_compounds']} / {M['drug_rows']:,}"], ['Condition knowledge base: compounds / with structure', f"164 / {N['kb1']['n_struct']}"]]
R = pd.read_csv(D + 'kb2_results.csv')
rows = []
lab = {'compound': 'New compound', 'dataset': 'New system', 'drug-external': 'New drug'}
for sch in ('compound', 'dataset', 'drug-external'):
    for m in ('Ridge', 'RF', 'LightGBM', 'ANN'):
        r = R[(R.scheme == sch) & (R.model == m)].iloc[0]
        rows.append([lab[sch], m, f"{r.MAE:.2f}", f"{r.RMSE:.2f}", f"{r.MedAE:.2f}", f"{r.R2:.3f}", f"{100*r.within5:.1f}"])
for m, l in (('Dataset mean', 'System-mean baseline'), ('LightGBM System only', 'LightGBM, system descriptors only'), ('LightGBM Molecular only', 'LightGBM, molecular descriptors only')):
    r = R[(R.scheme == 'compound') & (R.model == m)].iloc[0]
    rows.append(['New compound', l, f"{r.MAE:.2f}", f"{r.RMSE:.2f}", f"{r.MedAE:.2f}", f"{r.R2:.3f}", f"{100*r.within5:.1f}"])
T['t2'] = rows
CL = {'cardiovascular': 'Cardiovascular', 'cns': 'CNS', 'antibiotic': 'Anti-infective', 'analgesic': 'Analgesic', 'antidiabetic': 'Antidiabetic', 'other': 'Other', 'case study': 'Case-study drugs (not in name list)'}
E = pd.read_csv(D + 'drug_external.csv')
T['t3'] = [[CL.get(c, c), str(int(g.ik14.nunique())), str(len(g)), f"{g.err_lgb.mean():.2f}", f"{g.err_ann.mean():.2f}", f"{g.err_lgb.median():.2f}"] for c, g in E.groupby('cls')]
cs = pd.read_csv(D + 'case_summary.csv')
T['t4'] = [[r.drug, r.panel, (str(int(r.n)) if r.n == r.n else '0'), (str(int(r.datasets)) if r.n == r.n else '–'), (f"{r.mae_phi:.1f}" if r.n == r.n else '–'),
            (f"{r.mae_rt:.2f}" if r.n == r.n else '–'), (f"{r.med_rel:.1f}" if r.n == r.n else '–'), f"{r.ad_dist:.2f}"] for _, r in cs.iterrows()]
Sc = pd.read_csv(D + 'scouting.csv', dtype={'dataset': str})
rows = []
for p in ('A', 'B'):
    for _, s in Sc[Sc.panel == p].sort_values('score', ascending=False).head(4).iterrows():
        add = ', '.join(k for k, v in (('formic', s.formic), ('NH4 acetate', s.nh4ac), ('NH4 formate', s.nh4form)) if v > 0) or 'other/none'
        rows.append([p, s.dataset, s.column, f"{s.L:.0f} × {s.ID:.1f}", 'ACN' if s.acn_frac >= .5 else 'MeOH', add, f"{s.run:.1f}", f"{s.min_gap:.2f}", s.crit, str(s.url)])
T['t5'] = rows
# supplementary
cnt = P.groupby('dataset').size()
T['s1'] = [[r.dataset, str(r.column), r.usp, f"{r.L:.0f}×{r.ID:.1f}" if r.L == r.L and r.ID == r.ID else '', 'ACN' if r.acn_frac >= .5 else 'MeOH', f"{r.flow:.2f}" if r.flow == r.flow else '',
            f"{r['T']:.0f}" if r['T'] == r['T'] else '', str(int(cnt.get(r.dataset, 0))), str(r.url)] for _, r in S.iterrows()]
dp = pd.read_csv(D + 'drug_panel.csv')
T['s3'] = [[r['name'], CL[r.cls], str(r.n_obs), str(r.n_ds)] for _, r in dp.sort_values(['cls', 'name']).iterrows()]
st = pd.read_csv(D + 'kb1_structures.csv')
kb = pd.read_csv(D + 'knowledge_base_clean.csv').fillna('')
smi = st.set_index('compound').smiles.to_dict()
T['s2'] = [[str(i + 1), r['class'], r.compound, r.column, r.mobile_phase, r.buffer, str(r.flow), str(r.detection), smi.get(r.compound, '')] for i, (_, r) in enumerate(kb.iterrows())]
def cl(v):
    if isinstance(v, float) and v != v: return ''
    v = str(v); return '' if v in ('nan', 'None') else v
T = {k: [[cl(c) for c in row] for row in v] for k, v in T.items()}
json.dump(T, open(D + 'tables2.json', 'w'), ensure_ascii=False)
print({k: len(v) for k, v in T.items()})
