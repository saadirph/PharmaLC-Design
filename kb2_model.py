import pandas as pd, numpy as np, json, time, warnings, pickle, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
D = '/home/claude/data/'
t_start = time.time()
def log(*a):
    print(f'[{time.time()-t_start:6.0f}s]', *a, flush=True)

R = pd.read_csv(D + 'kb2_records.csv'); S = pd.read_csv(D + 'kb2_systems.csv', dtype={'dataset': str})
R['dataset'] = R.dataset.astype(str).str.zfill(4); S['dataset'] = S.dataset.astype(str).str.zfill(4)
X = pd.read_pickle(D + 'kb2_desc.pkl')
R['ik14'] = R['inchikey.std'].str[:14]
# molecular features
nanfrac = X.isna().mean(); molcols = [c for c in X.columns[1:] if nanfrac[c] < 0.05 and X[c].std() > 0]
X = X[['inchikey'] + molcols].rename(columns={'inchikey': 'inchikey.std'})
# system features
S['usp_L1'] = (S.usp == 'L1').astype(float); S['usp_L11'] = (S.usp == 'L11').astype(float); S['usp_other'] = (~S.usp.isin(['L1', 'L11'])).astype(float)
S['hsm_missing'] = S.H.isna().astype(float); S['tanaka_missing'] = S.kPB.isna().astype(float)
syscols = ['L', 'ID', 'dp', 'T', 'flow', 't0', 'phi0', 'phimax', 'slope', 'b', 'tG', 'acn_frac', 'pHA', 'H', 'Sst', 'A', 'B', 'C28', 'C70', 'kPB', 'aCH2', 'aTO', 'aBP',
           'usp_L1', 'usp_L11', 'usp_other', 'hsm_missing', 'tanaka_missing'] + [c for c in S.columns if c.startswith('add_')]
syscols = [c for c in syscols if S[c].notna().any() and S[c].std() > 0]
M = R.merge(X, on='inchikey.std', how='inner').merge(S[['dataset'] + syscols], on='dataset', how='left')
M = M.dropna(subset=['phi_e']).reset_index(drop=True)
y = M.phi_e.values
log('rows', len(M), 'mol feats', len(molcols), 'sys feats', len(syscols), 'compounds', M.ik14.nunique(), 'datasets', M.dataset.nunique())
FEATS = molcols + syscols
drugs = pd.read_csv(D + 'drug_panel.csv'); case = pd.read_csv(D + 'case_drugs.csv')
drug_ik = set(drugs.ik14) | set(case.inchikey.str[:14])
M['is_drug'] = M.ik14.isin(drug_ik)

def mk(name):
    if name == 'Ridge': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    if name == 'RF': return make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=120, max_features=0.33, min_samples_leaf=3, max_samples=0.35, n_jobs=2, random_state=0))
    if name == 'LightGBM': return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.04, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=2, verbose=-1, random_state=0)
    if name == 'ANN': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=1e-3, learning_rate_init=1e-3, batch_size=256, max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=12, random_state=0))

def metr(yt, yp):
    return dict(MAE=mean_absolute_error(yt, yp), RMSE=mean_squared_error(yt, yp) ** 0.5, R2=r2_score(yt, yp),
                MedAE=float(np.median(np.abs(yt - yp))), within5=float(np.mean(np.abs(yt - yp) <= 5)))

def clipX(Xm):
    return np.clip(Xm.astype(np.float64), -1e9, 1e9)

XA = clipX(M[FEATS].values)
results, oof = [], {}
MODELS = ['Ridge', 'RF', 'LightGBM', 'ANN']
for scheme, groups in (('compound', M.ik14.values), ('dataset', M.dataset.values)):
    gkf = GroupKFold(n_splits=5)
    for mname in MODELS:
        ck = D + f'ck_{scheme}_{mname}.npy'
        if os.path.exists(ck): pred = np.load(ck)
        else:
            pred = np.full(len(M), np.nan)
            for k, (tr, te) in enumerate(gkf.split(XA, y, groups)):
                m = mk(mname); m.fit(XA[tr], y[tr]); pred[te] = np.clip(m.predict(XA[te]), 0, 100)
            np.save(ck, pred)
        oof[(scheme, mname)] = pred
        r = metr(y, pred); r.update(scheme=scheme, model=mname)
        # per-fold-dataset median MAE for dataset scheme
        results.append(r); log(scheme, mname, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
# baseline: dataset mean (compound scheme)
base = M.groupby('dataset').phi_e.transform('mean').values
r = metr(y, base); r.update(scheme='compound', model='Dataset mean'); results.append(r)
# ablation: LightGBM mol only / sys only (compound scheme)
for lab, cols in (('Molecular only', molcols), ('System only', syscols)):
    Xs = clipX(M[cols].values); pred = np.full(len(M), np.nan)
    for tr, te in GroupKFold(5).split(Xs, y, M.ik14.values):
        m = mk('LightGBM'); m.fit(Xs[tr], y[tr]); pred[te] = np.clip(m.predict(Xs[te]), 0, 100)
    r = metr(y, pred); r.update(scheme='compound', model='LightGBM ' + lab); results.append(r); log(lab, round(r['MAE'], 2))
# learning curve (LightGBM, compound scheme, fixed test fold)
gkf = list(GroupKFold(5).split(XA, y, M.ik14.values)); tr0, te0 = gkf[0]
rng = np.random.default_rng(0); lc = []
for n in (500, 1000, 2500, 5000, 10000, 20000, len(tr0)):
    for rep in range(3 if n < len(tr0) else 1):
        idx = rng.choice(tr0, size=min(n, len(tr0)), replace=False)
        for mname in ('LightGBM', 'ANN'):
            m = mk(mname); m.fit(XA[idx], y[idx]); p = np.clip(m.predict(XA[te0]), 0, 100)
            lc.append(dict(n=n, rep=rep, model=mname, MAE=mean_absolute_error(y[te0], p)))
    log('lc', n)
# drug external test: train on non-drug compounds, test on drug compounds
tr_d = np.where(~M.is_drug)[0]; te_d = np.where(M.is_drug)[0]
drug_pred = {}
for mname in MODELS:
    m = mk(mname); m.fit(XA[tr_d], y[tr_d]); p = np.clip(m.predict(XA[te_d]), 0, 100); drug_pred[mname] = p
    r = metr(y[te_d], p); r.update(scheme='drug-external', model=mname); results.append(r); log('drug', mname, round(r['MAE'], 2), round(r['R2'], 3))
pd.DataFrame(results).to_csv(D + 'kb2_results.csv', index=False); pd.DataFrame(lc).to_csv(D + 'kb2_learning_curve.csv', index=False)
# final models on all data (for application + SHAP)
final = {}
for mname in ('LightGBM', 'ANN'):
    m = mk(mname); m.fit(XA, y); final[mname] = m
# applicability domain: PCA on scaled molecular descriptors, kNN distance (compound CV fold 0)
mol = clipX(M[molcols].values); imp = SimpleImputer(strategy='median').fit(mol); sc = StandardScaler().fit(imp.transform(mol))
Z = np.clip(sc.transform(imp.transform(mol)), -10, 10); pca = PCA(20, random_state=0).fit(Z); P = pca.transform(Z)
ad = np.full(len(M), np.nan)
for tr, te in GroupKFold(5).split(P, y, M.ik14.values):
    uq = np.unique(P[tr], axis=0)
    nn = NearestNeighbors(n_neighbors=5).fit(uq); dist, _ = nn.kneighbors(P[te]); ad[te] = dist.mean(1)

M['ad_dist'] = ad
for (s, mn), p in oof.items(): M[f'oof_{s}_{mn}'] = p
M.loc[te_d, 'drug_ext_LightGBM'] = drug_pred['LightGBM']; M.loc[te_d, 'drug_ext_ANN'] = drug_pred['ANN']
keep = ['id', 'name', 'dataset', 'ik14', 'rt', 'phi_e', 'is_drug', 'ad_dist', 'classyfire.superclass', 'MolLogP', 'TPSA', 'MolWt'] + [c for c in M.columns if c.startswith('oof_') or c.startswith('drug_ext')]
M[keep].to_csv(D + 'kb2_predictions.csv', index=False)
pd.DataFrame(results).to_csv(D + 'kb2_results.csv', index=False); pd.DataFrame(lc).to_csv(D + 'kb2_learning_curve.csv', index=False)
pickle.dump(dict(final=final, FEATS=FEATS, molcols=molcols, syscols=syscols, imp=imp, sc=sc, pca=pca, Ptrain=P), open(D + 'kb2_models.pkl', 'wb'))
# SHAP for LightGBM
import shap
idx = np.random.default_rng(1).choice(len(M), 4000, replace=False)
ex = shap.TreeExplainer(final['LightGBM']); sv = ex.shap_values(XA[idx])
np.save(D + 'kb2_shap.npy', sv); np.save(D + 'kb2_shap_idx.npy', idx)
json.dump(dict(rows=len(M), compounds=int(M.ik14.nunique()), datasets=int(M.dataset.nunique()), n_mol=len(molcols), n_sys=len(syscols),
               drug_rows=int(M.is_drug.sum()), drug_compounds=int(M[M.is_drug].ik14.nunique())), open(D + 'kb2_meta.json', 'w'))
log('done')
