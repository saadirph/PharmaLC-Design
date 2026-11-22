import pandas as pd, numpy as np, pickle, json, warnings
warnings.filterwarnings('ignore')
D = '/home/claude/data/'; BASE = '/home/claude/ext/RepoRT/processed_data'
ORG = ['meoh', 'acn', 'iproh', 'acetone']
mods = pickle.load(open(D + 'kb2_models.pkl', 'rb'))
FEATS, molcols, syscols = mods['FEATS'], mods['molcols'], mods['syscols']
S = pd.read_csv(D + 'kb2_systems.csv', dtype={'dataset': str}); S['dataset'] = S.dataset.str.zfill(4)
for c, v in (('usp_L1', S.usp == 'L1'), ('usp_L11', S.usp == 'L11'), ('usp_other', ~S.usp.isin(['L1', 'L11'])), ('hsm_missing', S.H.isna()), ('tanaka_missing', S.kPB.isna())):
    S[c] = v.astype(float)
X = pd.read_pickle(D + 'kb2_desc.pkl')
case = pd.read_csv(D + 'case_drugs.csv'); P = pd.read_csv(D + 'kb2_predictions.csv', dtype={'dataset': str}); P['dataset'] = P.dataset.str.zfill(4)

def grad(ds):
    md = pd.read_csv(f'{BASE}/{ds}/{ds}_metadata.tsv', sep='\t').iloc[0].to_dict()
    gr = pd.read_csv(f'{BASE}/{ds}/{ds}_gradient.tsv', sep='\t')
    t = pd.to_numeric(gr.iloc[:, 0], errors='coerce').values; comp = np.zeros(len(gr))
    for E in 'ABCD':
        org = sum(float(md.get(f'eluent.{E}.{o}', 0) or 0) for o in ORG)
        col = [c for c in gr.columns if c.strip().startswith(E + ' ')]
        if col: comp += pd.to_numeric(gr[col[0]], errors='coerce').fillna(0).values / 100 * org
    return t, comp

def phi_to_t(ds, phi):
    t, comp = grad(ds); t0 = float(S.set_index('dataset').loc[ds, 't0'])
    # first ramp segment region up to max
    kmax = int(np.argmax(comp)); tt, cc = t[:kmax + 1], comp[:kmax + 1]
    phi = np.clip(phi, cc.min(), cc.max())
    # monotone increasing part: use cumulative max
    cm = np.maximum.accumulate(cc)
    out = np.interp(phi, cm, tt)
    return out + t0

def predict(ik_list, ds_list):
    rows = []
    for ik, ds in zip(ik_list, ds_list):
        rows.append(np.concatenate([X.set_index('inchikey').loc[ik, molcols].values.astype(float), S.set_index('dataset').loc[ds, syscols].values.astype(float)]))
    A = np.clip(np.array(rows, dtype=float), -1e9, 1e9)
    return {k: m.predict(A) for k, m in mods['final'].items()}

def ad(ik):
    v = np.clip(X.set_index('inchikey').loc[[ik], molcols].values.astype(float), -1e9, 1e9)
    z = np.clip(mods['sc'].transform(mods['imp'].transform(v)), -10, 10); p = mods['pca'].transform(z)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5).fit(np.unique(mods['Ptrain'], axis=0))
    return nn.kneighbors(p)[0].mean()

# 1. Literature check: case drugs, out-of-fold (compound-grouped) predictions
P['ik14'] = P.ik14.astype(str)
lit = []
for _, c in case.iterrows():
    sub = P[P.ik14 == c.inchikey[:14]]
    for _, r in sub.iterrows():
        ens = np.nanmean([r['oof_compound_LightGBM'], r['oof_compound_ANN']])
        try: tpred = float(phi_to_t(r.dataset, np.array([ens]))[0])
        except Exception: tpred = np.nan
        sysr = S.set_index('dataset').loc[r.dataset]
        lit.append(dict(drug=c.drug, panel=c.panel, dataset=r.dataset, column=sysr.column, url=sysr.url, rt_lit=r.rt, phi_lit=r.phi_e,
                        phi_ann=r['oof_compound_ANN'], phi_lgb=r['oof_compound_LightGBM'], phi_ens=ens, rt_pred=tpred, t_run=sysr.tG + sysr.t0))
L = pd.DataFrame(lit); L['abs_err_phi'] = (L.phi_ens - L.phi_lit).abs(); L['abs_err_rt'] = (L.rt_pred - L.rt_lit).abs(); L['rel_err_run'] = 100 * L.abs_err_rt / L.t_run
L.to_csv(D + 'case_literature_check.csv', index=False)
summ = L.groupby('drug').agg(n=('rt_lit', 'size'), datasets=('dataset', 'nunique'), mae_phi=('abs_err_phi', 'mean'), mae_rt=('abs_err_rt', 'mean'),
                              med_rel=('rel_err_run', 'median')).reset_index()
case['ad_dist'] = [ad(ik) for ik in case.inchikey]
summ = case[['drug', 'panel', 'ad_dist']].merge(summ, on='drug', how='left')
summ.to_csv(D + 'case_summary.csv', index=False); print(summ.round(2))

# 2. In-silico scouting over candidate literature systems
cand = S[(S.usp == 'L1') & S.H.notna()].copy()
cnt = P.groupby('dataset').size(); cand['n'] = cand.dataset.map(cnt)
cand = cand.sort_values('n', ascending=False)
cand = cand.drop_duplicates(subset=['column', 'acn_frac', 'add_formic', 'add_nh4ac', 'add_nh4form']).head(20)
res = []; prof = []
for panel in ('A', 'B'):
    dr = case[case.panel == panel]
    for _, s in cand.iterrows():
        pr = predict(dr.inchikey.tolist(), [s.dataset] * len(dr))
        ens = (pr['LightGBM'] + pr['ANN']) / 2
        tr = phi_to_t(s.dataset, ens)
        o = np.argsort(tr); gaps = np.diff(tr[o]); k = int(np.argmin(gaps)) if len(gaps) else 0
        res.append(dict(panel=panel, dataset=s.dataset, column=s.column, acn_frac=s.acn_frac, formic=s.add_formic, nh4ac=s.add_nh4ac, nh4form=s.add_nh4form,
                        flow=s.flow, L=s.L, ID=s.ID, T=s['T'], run=s.tG + s.t0, min_gap=gaps.min(), crit=f"{dr.drug.values[o][k]} / {dr.drug.values[o][k+1]}",
                        span=tr.max() - tr.min(), min_dphi=np.diff(np.sort(ens)).min(), url=s.url))
        for d, t, p in zip(dr.drug, tr, ens): prof.append(dict(panel=panel, dataset=s.dataset, drug=d, rt=t, phi=p))
Rz = pd.DataFrame(res); Rz['score'] = Rz.min_gap / Rz.run
Rz.to_csv(D + 'scouting.csv', index=False); pd.DataFrame(prof).to_csv(D + 'scouting_profiles.csv', index=False)
print(Rz.sort_values(['panel', 'score'], ascending=[True, False]).groupby('panel').head(3)[['panel', 'dataset', 'column', 'acn_frac', 'min_gap', 'crit', 'run']])

# 3. Drug panel by therapeutic class (external drug test)
dp = pd.read_csv(D + 'drug_panel.csv'); P2 = P[P.is_drug].copy()
P2 = P2.merge(dp[['ik14', 'cls', 'name']].rename(columns={'name': 'drug'}), on='ik14', how='left')
P2['cls'] = P2.cls.fillna('case study')
P2['err_lgb'] = (P2.drug_ext_LightGBM - P2.phi_e).abs(); P2['err_ann'] = (P2.drug_ext_ANN - P2.phi_e).abs()
P2.to_csv(D + 'drug_external.csv', index=False)
print(P2.groupby('cls')[['err_lgb', 'err_ann']].agg(['mean', 'size']).round(2))
