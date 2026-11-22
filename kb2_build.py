"""Build the literature knowledge base from RepoRT (gradient RP datasets) with system features and elution-composition target."""
import pandas as pd, numpy as np, glob, os, re, json
BASE = '/home/claude/ext/RepoRT/processed_data'
RES = '/home/claude/ext/RepoRT/resources'
rng = np.random.default_rng(42)
ORG = ['meoh', 'acn', 'iproh', 'acetone']
ADD = ['formic', 'acetic', 'trifluoroacetic', 'phosphor', 'nh4ac', 'nh4form', 'nh4carb', 'nh4bicarb', 'nh4f', 'nh4oh', 'trieth']
CAP = 2500

hsm = pd.read_csv(f'{RES}/hsm_database/hsm_database.tsv', sep='\t')
hsm['key'] = hsm.name_normalized.str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
tan = pd.read_csv(f'{RES}/tanaka_database/tanaka_database.tsv', sep='\t')
tan['key'] = tan.name_normalized.str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

def phi_fun(md, gr):
    org = {}
    for E in 'ABCD':
        org[E] = sum(float(md.get(f'eluent.{E}.{o}', 0) or 0) for o in ORG)
    t = pd.to_numeric(gr.iloc[:, 0], errors='coerce').values
    comp = np.zeros(len(gr))
    for i, E in enumerate('ABCD'):
        col = [c for c in gr.columns if c.strip().startswith(E + ' ')]
        if col: comp += pd.to_numeric(gr[col[0]], errors='coerce').fillna(0).values / 100 * org[E]
    return t, comp, org

systems, records = [], []
for d in sorted(glob.glob(f'{BASE}/[0-9]*')):
    i = os.path.basename(d)
    try:
        info = pd.read_csv(f'{d}/{i}_info.tsv', sep='\t', dtype=str).iloc[0]
        if info['method.type'] != 'RP': continue
        md = pd.read_csv(f'{d}/{i}_metadata.tsv', sep='\t').iloc[0].to_dict()
        gr = pd.read_csv(f'{d}/{i}_gradient.tsv', sep='\t')
        rt = pd.read_csv(f'{d}/{i}_rtdata_canonical_success.tsv', sep='\t')
    except Exception:
        continue
    if len(gr) < 2 or len(rt) < 20: continue
    t, comp, org = phi_fun(md, gr)
    if np.nanmax(comp) - np.nanmin(comp) < 5: continue  # isocratic / no gradient
    t0 = float(md.get('column.t0') or np.nan)
    L, ID = md.get('column.length'), md.get('column.id')
    flow = md.get('column.flowrate')
    if not (t0 > 0): continue
    # first ramp
    k = np.argmax(np.diff(comp) > 0)
    slope = (comp[k + 1] - comp[k]) / max(t[k + 1] - t[k], 1e-6)
    tend = t[np.argmax(comp)]  # time max organic reached
    key = re.sub(r'[^a-z0-9]', '', str(md.get('column.name', '')).lower())
    h = hsm[hsm.key == key]; ta = tan[tan.key == key]
    accB = float(md.get('eluent.B.acn', 0) or 0); meB = float(md.get('eluent.B.meoh', 0) or 0)
    sysf = dict(dataset=i, ds_name=info['name'], url=info.get('url'), authors=info.get('authors'),
                column=md.get('column.name'), usp=md.get('column.usp.code'), L=L, ID=ID, dp=md.get('column.particle.size'),
                T=md.get('column.temperature'), flow=flow, t0=t0, phi0=comp[0], phimax=np.nanmax(comp), slope=slope,
                b=slope * t0, tG=tend, acn_frac=accB / max(accB + meB, 1e-6), n_raw=len(rt),
                pHA=md.get('eluent.A.pH'),
                H=h.H.iloc[0] if len(h) else np.nan, Sst=h['S*'].iloc[0] if len(h) else np.nan, A=h.A.iloc[0] if len(h) else np.nan,
                B=h.B.iloc[0] if len(h) else np.nan, C28=h['C (pH 2.8)'].iloc[0] if len(h) else np.nan, C70=h['C (pH 7.0)'].iloc[0] if len(h) else np.nan,
                kPB=ta.kPB.iloc[0] if len(ta) else np.nan, aCH2=ta['αCH2'].iloc[0] if len(ta) else np.nan,
                aTO=ta['αT/O'].iloc[0] if len(ta) else np.nan, aBP=ta['αB/P'].iloc[0] if len(ta) else np.nan)
    for a in ADD:
        sysf['add_' + a] = float(md.get(f'eluent.A.{a}', 0) or 0) + float(md.get(f'eluent.B.{a}', 0) or 0)
    systems.append(sysf)
    rt = rt[['id', 'name', 'formula', 'rt', 'smiles.std', 'inchikey.std', 'classyfire.superclass', 'classyfire.class']].dropna(subset=['rt', 'smiles.std'])
    rt = rt[rt.rt > 0]
    if len(rt) > CAP: rt = rt.sample(CAP, random_state=1)
    te = rt.rt.values - t0
    phi_e = np.interp(np.clip(te, t[0], t[-1]), t, comp)
    rt = rt.assign(dataset=i, phi_e=phi_e, void=rt.rt.values < 1.2 * t0, late=rt.rt.values - t0 > tend)
    records.append(rt)
S = pd.DataFrame(systems); R = pd.concat(records)
for c in ['L', 'ID', 'dp', 'T', 'flow', 'pHA']: S[c] = pd.to_numeric(S[c], errors='coerce')
print('systems', len(S), 'records', len(R), 'unique compounds', R['inchikey.std'].nunique())
print('void', R.void.mean().round(3), 'late', R.late.mean().round(3))
R = R[~R.void & ~R.late]
# keep datasets with >= 20 usable compounds
cnt = R.groupby('dataset').size(); keep = cnt[cnt >= 20].index
R = R[R.dataset.isin(keep)]; S = S[S.dataset.isin(keep)]
print('after filtering: systems', len(S), 'records', len(R), 'unique', R['inchikey.std'].nunique())
S.to_csv('/home/claude/data/kb2_systems.csv', index=False); R.to_csv('/home/claude/data/kb2_records.csv', index=False)
print(S.usp.value_counts().head(), S[['H', 'kPB']].notna().mean())
