#!/usr/bin/env python3
"""
Adds Leave-Publication-Out (LPO), Bemis-Murcko scaffold and MCES-10 splits to the
existing kb2_model.py pipeline. Feature assembly, models (mk) and metrics are copied
from kb2_model.py so results are comparable with Table S5 (GroupKFold(5), same hyperparameters).

STATUS: LPO group/fold generation tested on kb2_records.csv + kb2_systems.csv.
NOT tested (no RDKit / LightGBM / kb2_desc.pkl in the authoring sandbox): scaffold, MCES, training.
REQUIRES for training: kb2_desc.pkl (output of kb2_feat.py), lightgbm, scikit-learn.
REQUIRES for scaffold: rdkit. For MCES: a TSV of pair distances from the myopic-mces package
(columns smiles_a, smiles_b, distance; check that package's README for its command line).

  python kb2_splits.py --data Code_and_data/ --splits lpo --splits-only
  python kb2_splits.py --data Code_and_data/ --splits lpo scaffold mces --mces-pairs pairs.tsv --out splits_out/
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')


def load_base(D):
    R = pd.read_csv(D + 'kb2_records.csv'); S = pd.read_csv(D + 'kb2_systems.csv', dtype={'dataset': str})
    R['dataset'] = R.dataset.astype(str).str.zfill(4); S['dataset'] = S.dataset.astype(str).str.zfill(4)
    R['ik14'] = R['inchikey.std'].str[:14]
    # publication group: DOI/URL when present, otherwise each direct submission is its own group
    S['pub'] = np.where(S.url.notna(), S.url, 'direct_' + S.dataset)
    return R, S


def split_groups(R, S, name, mces_pairs=None):
    if name == 'lpo':
        return R.dataset.map(S.set_index('dataset')['pub'])
    if name == 'scaffold':
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        cache = {}
        def f(smi):
            if smi not in cache:
                m = Chem.MolFromSmiles(smi)
                cache[smi] = 'INVALID' if m is None else (MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'ACYCLIC')
            return cache[smi]
        return R['smiles.std'].map(f)
    if name == 'mces':
        import networkx as nx
        pr = pd.read_csv(mces_pairs, sep='\t'); G = nx.Graph(); G.add_nodes_from(R['smiles.std'].unique())
        G.add_edges_from(pr.loc[pr.distance <= 10, ['smiles_a', 'smiles_b']].itertuples(index=False, name=None))
        comp = {n: f'c{i}' for i, c in enumerate(nx.connected_components(G)) for n in c}
        g = R['smiles.std'].map(comp); top = g.value_counts(normalize=True).iloc[0]
        print(f'[MCES] components={g.nunique()}; largest = {top:.1%} of records', file=sys.stderr)
        if top > 0.4: print('[MCES] WARNING: giant component; decide a handling rule before using this split.', file=sys.stderr)
        return g
    raise ValueError(name)


def audit(R, folds):
    seen = []
    for k, (tr, te) in enumerate(folds):
        seen.append(float(R.iloc[te].ik14.isin(set(R.iloc[tr].ik14)).mean()))
    return seen


def build_features(D, R, S):
    X = pd.read_pickle(D + 'kb2_desc.pkl')
    nanfrac = X.isna().mean(); molcols = [c for c in X.columns[1:] if nanfrac[c] < 0.05 and X[c].std() > 0]
    X = X[['inchikey'] + molcols].rename(columns={'inchikey': 'inchikey.std'})
    S = S.copy()
    S['usp_L1'] = (S.usp == 'L1').astype(float); S['usp_L11'] = (S.usp == 'L11').astype(float); S['usp_other'] = (~S.usp.isin(['L1', 'L11'])).astype(float)
    S['hsm_missing'] = S.H.isna().astype(float); S['tanaka_missing'] = S.kPB.isna().astype(float)
    syscols = ['L', 'ID', 'dp', 'T', 'flow', 't0', 'phi0', 'phimax', 'slope', 'b', 'tG', 'acn_frac', 'pHA', 'H', 'Sst', 'A', 'B', 'C28', 'C70', 'kPB', 'aCH2', 'aTO', 'aBP',
               'usp_L1', 'usp_L11', 'usp_other', 'hsm_missing', 'tanaka_missing'] + [c for c in S.columns if c.startswith('add_')]
    syscols = [c for c in syscols if S[c].notna().any() and S[c].std() > 0]
    M = R.merge(X, on='inchikey.std', how='inner').merge(S[['dataset'] + syscols], on='dataset', how='left')
    M = M.dropna(subset=['phi_e']).reset_index(drop=True)
    return M, molcols + syscols


def mk(name):  # identical to kb2_model.py
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import lightgbm as lgb
    if name == 'Ridge': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    if name == 'RF': return make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=120, max_features=0.33, min_samples_leaf=3, max_samples=0.35, n_jobs=2, random_state=0))
    if name == 'LightGBM': return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.04, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=2, verbose=-1, random_state=0)
    if name == 'ANN': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=1e-3, learning_rate_init=1e-3, batch_size=256, max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=12, random_state=0))


def metr(yt, yp):
    e = yt - yp
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), R2=float(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
                MedAE=float(np.median(np.abs(e))), within5=float((np.abs(e) <= 5).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True)
    ap.add_argument('--splits', nargs='+', default=['lpo'], choices=['lpo', 'scaffold', 'mces'])
    ap.add_argument('--models', nargs='+', default=['Ridge', 'RF', 'LightGBM', 'ANN'])
    ap.add_argument('--mces-pairs'); ap.add_argument('--splits-only', action='store_true'); ap.add_argument('--out', default='splits_out')
    a = ap.parse_args(); D = a.data if a.data.endswith('/') else a.data + '/'; os.makedirs(a.out, exist_ok=True)
    R, S = load_base(D)
    if a.splits_only:
        M, FEATS = R.dropna(subset=['phi_e']).reset_index(drop=True), None
    else:
        M, FEATS = build_features(D, R, S)
    y = M.phi_e.values; summary, assign = {}, M[['id', 'dataset', 'ik14']].copy() if 'id' in M else M[['dataset', 'ik14']].copy()
    for sp in a.splits:
        g = split_groups(M, S, sp, a.mces_pairs).values
        folds = list(GroupKFold(5).split(np.zeros(len(M)), y, g))
        fold = np.zeros(len(M), int)
        for k, (_, te) in enumerate(folds): fold[te] = k
        assign[f'group_{sp}'], assign[f'fold_{sp}'] = g, fold
        info = dict(n_groups=int(pd.Series(g).nunique()), fold_records=[int(len(te)) for _, te in folds],
                    test_records_with_compound_seen_in_train=[round(x, 3) for x in audit(M, folds)])
        if not a.splits_only:
            XA = np.clip(M[FEATS].values.astype(np.float64), -1e9, 1e9); info['metrics'] = {}
            for mn in a.models:
                pred = np.full(len(M), np.nan)
                for tr, te in folds:
                    m = mk(mn); m.fit(XA[tr], y[tr]); pred[te] = np.clip(m.predict(XA[te]), 0, 100)
                info['metrics'][mn] = metr(y, pred); print(sp, mn, info['metrics'][mn], flush=True)
                assign[f'oof_{sp}_{mn}'] = pred
        summary[sp] = info; print(sp, json.dumps({k: v for k, v in info.items() if k != 'metrics'}))
    assign.to_csv(os.path.join(a.out, 'split_assignments.csv'), index=False)
    json.dump(summary, open(os.path.join(a.out, 'split_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
