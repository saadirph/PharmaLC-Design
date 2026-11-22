import pandas as pd, numpy as np, json
D = '/home/claude/data/'
N = {}
N['meta'] = json.load(open(D + 'kb2_meta.json'))
R = pd.read_csv(D + 'kb2_results.csv')
N['res'] = {f"{r.scheme}|{r.model}": dict(MAE=r.MAE, RMSE=r.RMSE, R2=r.R2, MedAE=r.MedAE, w5=r.within5) for _, r in R.iterrows()}
LC = pd.read_csv(D + 'kb2_learning_curve.csv'); N['lc'] = LC.groupby(['model', 'n']).MAE.mean().unstack(0).round(2).reset_index().to_dict('records')
N['interp'] = json.load(open(D + 'interp.json'))
L = pd.read_csv(D + 'case_literature_check.csv')
N['case'] = dict(n=len(L), systems=int(L.dataset.nunique()), mae_rt=L.abs_err_rt.mean(), med_rt=L.abs_err_rt.median(), mae_phi=L.abs_err_phi.mean(),
                 med_rel=L.rel_err_run.median(), within1=float((L.abs_err_rt <= 1).mean()))
N['case_summary'] = pd.read_csv(D + 'case_summary.csv').round(3).fillna('').to_dict('records')
Sc = pd.read_csv(D + 'scouting.csv', dtype={'dataset': str})
N['scout'] = {p: Sc[Sc.panel == p].sort_values('score', ascending=False).head(4).round(3).to_dict('records') for p in ('A', 'B')}
N['scout_n'] = int(Sc.dataset.nunique())
N['scout_frac_ok'] = {p: float((Sc[Sc.panel == p].min_gap >= 0.5).mean()) for p in ('A', 'B')}
E = pd.read_csv(D + 'drug_external.csv')
N['drug_cls'] = E.groupby('cls').agg(n=('err_lgb', 'size'), lgb=('err_lgb', 'mean'), ann=('err_ann', 'mean')).round(2).reset_index().to_dict('records')
N['kb1'] = json.load(open(D + 'kb1_struct_summary.json')); N['kb'] = json.load(open(D + 'summary.json'))['kb_stats']
S = pd.read_csv(D + 'kb2_systems.csv'); P = pd.read_csv(D + 'kb2_predictions.csv')
N['sys'] = dict(n=len(S), hsm=int(S.H.notna().sum()), tanaka=int(S.kPB.notna().sum()), acn=int((S.acn_frac >= .5).sum()), L1=int((S.usp == 'L1').sum()),
                L11=int((S.usp == 'L11').sum()), formic=int((S.add_formic > 0).sum()), flow_med=S.flow.median(), t0_med=S.t0.median(), cols=int(S.column.nunique()),
                studies=int(S.url.nunique()))
per = P.assign(e=(P.oof_dataset_LightGBM - P.phi_e).abs()).groupby('dataset').e.mean()
N['per_sys'] = dict(median=per.median(), q1=per.quantile(.25), q3=per.quantile(.75), lt5=float((per < 5).mean()))
s = json.dumps(N, indent=1, default=float).replace('NaN', 'null'); open(D + 'numbers.json', 'w').write(s)
print(json.dumps({k: N[k] for k in ['meta', 'case', 'sys', 'per_sys', 'scout_frac_ok']}, indent=1, default=float))
for k, v in N['res'].items(): print(k, {a: round(b, 3) for a, b in v.items()})
