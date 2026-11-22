import sys, json, math, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/scripts')
import figs as F0  # shared style
from figs import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score
D = '/home/claude/data/'
OUTD = '/home/claude/figs2/'
import os; os.makedirs(OUTD, exist_ok=True)
F0.OUTD = OUTD
GREY = '#b9b8b2'
CLS_ORDER = ['cardiovascular', 'cns', 'antibiotic', 'analgesic', 'antidiabetic', 'other']
CLS_LAB = {'cardiovascular': 'Cardiovascular', 'cns': 'CNS', 'antibiotic': 'Anti-infective', 'analgesic': 'Analgesic', 'antidiabetic': 'Antidiabetic', 'other': 'Other'}

def figW():
    fig = plt.figure(figsize=(W2, 52 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 52); ax.axis('off')
    def box(x, y, w, h, title, body, fc='#ffffff', ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fc, ec=ec, lw=0.6))
        ax.text(x + 2, y + h - 2.4, title, fontsize=6.3, fontweight='bold', va='top')
        ax.text(x + 2, y + h - 7, body, fontsize=5.2, va='top', color=INK2, linespacing=1.35)
    def arrow(x0, y0, x1, y1):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK2, mutation_scale=6, shrinkA=0, shrinkB=0))
    meta = json.load(open(D + 'kb2_meta.json'))
    box(1, 27, 34, 23, 'Literature sources', f"RepoRT: {meta['datasets']} gradient RP\nsystems, published studies\nSIELC application notes:\n164 compounds", fc='#f4f8fd', ec=BLUE)
    box(1, 2, 34, 22, 'Curation', 'Structure standardisation\n(InChIKey), dead-time and\nlate-elution filtering,\nelution composition φ$_e$')
    box(42, 14, 34, 36, 'Featurisation', f"{meta['n_mol']} RDKit descriptors\n{meta['n_sys']} system descriptors:\ncolumn (HSM, Tanaka),\ngeometry, T, flow, t$_0$,\nmodifier, additives,\ngradient φ$_0$, slope, b")
    box(83, 14, 34, 36, 'Learning', 'ANN (MLP 256-128-64)\nLightGBM, random forest,\nridge baseline\n\nValidation: new compound,\nnew system, new drug\nSHAP, applicability domain')
    box(124, 30, 57, 20, 'Application: drug panels', f"{meta['drug_compounds']} drugs in 6 therapeutic classes\nCase study: 7 cardiometabolic + 3 CNS\ndrugs vs published retention data", fc='#fdf1ec', ec=ORANGE)
    box(124, 6, 57, 20, 'In-silico method scouting', 'Predicted t$_R$ of each panel in\n20 candidate C18 systems;\ncritical pair and minimum gap', fc='#fdf1ec', ec=ORANGE)
    arrow(18, 27, 18, 24); arrow(35, 13, 42, 22); arrow(76, 32, 83, 32); arrow(117, 38, 124, 40); arrow(117, 26, 124, 18)
    save(fig, 'Fig1_workflow')

def figKB():
    X = pd.read_pickle(D + 'kb2_desc.pkl'); mods = pickle.load(open(D + 'kb2_models.pkl', 'rb'))
    S = pd.read_csv(D + 'kb2_systems.csv', dtype={'dataset': str}); P = pd.read_csv(D + 'kb2_predictions.csv', dtype={'dataset': str})
    dp = pd.read_csv(D + 'drug_panel.csv'); case = pd.read_csv(D + 'case_drugs.csv')
    v = np.clip(X[mods['molcols']].values.astype(float), -1e9, 1e9)
    Z = np.clip(mods['sc'].transform(mods['imp'].transform(v)), -10, 10); PC = mods['pca'].transform(Z)[:, :2]
    ik14 = X.inchikey.str[:14].values
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.5, left=0.07, right=0.985, top=0.95, bottom=0.08, width_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.hexbin(PC[:, 0], PC[:, 1], gridsize=70, cmap=cmap_seq, bins='log', mincnt=1, lw=0, extent=(-12, 25, -14, 16))
    isd = np.isin(ik14, dp.ik14.values)
    ax.scatter(PC[isd, 0], PC[isd, 1], s=5, color=ORANGE, ec='white', lw=0.25, zorder=3, label=f'Drugs (n = {isd.sum()})')
    for _, c in case.iterrows():
        j = np.where(ik14 == c.inchikey[:14])[0]
        if len(j):
            ax.scatter(PC[j[0], 0], PC[j[0], 1], s=16, color='white', ec=INK, lw=0.8, zorder=4)
            OFF = {'Alprazolam': (-4, 6, 'right'), 'Bromazepam': (5, -5, 'left'), 'Clopidogrel': (-5, -6, 'right'), 'Amlodipine': (5, 3, 'left'),
                   'Bisoprolol': (5, 3, 'left'), 'Methylphenidate': (-5, -6, 'right'), 'Clarithromycin': (-5, 5, 'right'), 'Isosorbide mononitrate': (5, -5, 'left')}
            dx, dy, ha = OFF.get(c.drug, (4, 2, 'left'))
            ax.annotate(c.drug, (PC[j[0], 0], PC[j[0], 1]), xytext=(dx, dy), textcoords='offset points', fontsize=4.8, color=INK, ha=ha,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
    ev = mods['pca'].explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({100*ev[0]:.0f}% variance)'); ax.set_ylabel(f'PC2 ({100*ev[1]:.0f}%)'); ax.set_xlim(-12, 25); ax.set_ylim(-14, 16)
    ax.legend(handles=[Patch(fc=SEQ[4], label=f'All compounds (n = {len(PC):,}), density'), Line2D([], [], marker='o', ls='', color=ORANGE, ms=3, label=f'Drug panel (n = {isd.sum()})'),
                       Line2D([], [], marker='o', ls='', mfc='white', mec=INK, ms=4, label='Case-study drugs')], loc='upper right')
    ax.set_title('Chemical space (RDKit descriptors, PCA)', pad=4); label(ax, 'a', x=-0.1, y=1.0)
    ax = fig.add_subplot(gs[0, 1])
    s2 = S[S.H.notna()]; n = P.groupby('dataset').size(); s2 = s2.assign(n=s2.dataset.str.zfill(4).map(n.rename(lambda k: str(k).zfill(4))).fillna(20))
    acn = s2.acn_frac >= 0.5
    ax.scatter(s2.H[acn], s2.C28[acn], s=np.sqrt(s2.n[acn]) * 1.5, color=BLUE, alpha=0.6, ec='white', lw=0.3, label='Acetonitrile')
    ax.scatter(s2.H[~acn], s2.C28[~acn], s=np.sqrt(s2.n[~acn]) * 1.5, color=ORANGE, alpha=0.7, ec='white', lw=0.3, label='Methanol')
    ax.set_xlabel('HSM hydrophobicity, H'); ax.set_ylabel('HSM cation exchange, C (pH 2.8)'); ax.legend(loc='upper left', title=None)
    ax.set_title(f'Column selectivity (n = {len(s2)})', pad=4); label(ax, 'b', x=-0.25)
    ax = fig.add_subplot(gs[0, 2])
    adds = {'Formic acid': 'add_formic', 'Ammonium acetate': 'add_nh4ac', 'Ammonium formate': 'add_nh4form', 'Acetic acid': 'add_acetic', 'TFA': 'add_trifluoroacetic',
            'Phosphoric acid': 'add_phosphor', 'NH$_4$OH / NH$_4$HCO$_3$': None}
    cnts = {}
    for k, c in adds.items():
        cnts[k] = int((S[c] > 0).sum()) if c else int(((S.get('add_nh4oh', 0) > 0) | (S.get('add_nh4bicarb', 0) > 0)).sum())
    kk = list(cnts); vv = [cnts[k] for k in kk]
    ax.barh(range(len(kk)), vv, color=BLUE, height=0.6, lw=0)
    for i, x in enumerate(vv): ax.text(x + 2, i, str(x), va='center', fontsize=5, color=INK2)
    ax.set_yticks(range(len(kk)), kk); ax.invert_yaxis(); ax.set_xlabel('Systems (n)'); ax.tick_params(axis='y', length=0)
    ax.set_title('Mobile-phase additives', pad=4); label(ax, 'c', x=-0.55)
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(P.phi_e, bins=40, color=BLUE, lw=0); ax.set_xlabel('Elution composition φ$_e$ (% organic)'); ax.set_ylabel('Records (n)')
    ax.set_title(f'Target distribution (n = {len(P):,})', pad=4); label(ax, 'd', x=-0.25)
    ax = fig.add_subplot(gs[1, 2])
    g = dp.groupby('cls').agg(n=('name', 'size'), obs=('n_obs', 'sum')).reindex(CLS_ORDER)
    y = np.arange(len(g))
    ax.barh(y - 0.2, g.n, height=0.38, color=ORANGE, lw=0, label='Drugs')
    ax.barh(y + 0.2, g.obs / 5, height=0.38, color='white', ec=ORANGE, lw=0.6, label='Records ÷ 5')
    for i, (a, b) in enumerate(zip(g.n, g.obs)):
        ax.text(a + 1, i - 0.2, str(a), va='center', fontsize=5, color=INK2); ax.text(b / 5 + 1, i + 0.2, str(b), va='center', fontsize=5, color=MUTED)
    ax.set_yticks(y, [CLS_LAB[c] for c in g.index]); ax.invert_yaxis(); ax.tick_params(axis='y', length=0); ax.set_xlabel('Count')
    ax.legend(loc='lower left', bbox_to_anchor=(0.35, 1.0), ncol=2, handlelength=1); ax.set_title('Drug panel by class', pad=14); label(ax, 'e', x=-0.55)
    save(fig, 'Fig2_literature_knowledge_base')

def figSIELC():
    kb = pd.read_csv(D + 'knowledge_base_clean.csv'); st = pd.read_csv(D + 'kb1_structures.csv'); ss = json.load(open(D + 'kb1_struct_summary.json'))
    S = json.load(open(D + 'summary.json'))['kb_stats']
    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.62, hspace=0.6, left=0.13, right=0.985, top=0.94, bottom=0.09, width_ratios=[1.15, 1.15, 1])
    cls = ['Hydrophobic', 'Hydrophilic']; col = {'Hydrophobic': BLUE, 'Hydrophilic': ORANGE}
    def grouped(ax, var, order, labels, title, stat):
        ct = pd.crosstab(kb[var].fillna('No additive'), kb['class']).reindex(order).fillna(0)
        yy = np.arange(len(order)); h = 0.36
        for i, c in enumerate(cls):
            ax.barh(yy + (i - 0.5) * (h + 0.04), ct[c], height=h, color=col[c], lw=0, label=c)
            for y0, v in zip(yy + (i - 0.5) * (h + 0.04), ct[c]):
                if v > 0: ax.text(v + 0.8, y0, f'{int(v)}', va='center', fontsize=4.8, color=INK2)
        ax.set_yticks(yy, labels); ax.invert_yaxis(); ax.set_xlabel('Compounds (n)'); ax.tick_params(axis='y', length=0)
        ax.set_title(title, pad=10); ax.text(1.0, 1.01, stat, transform=ax.transAxes, ha='right', va='bottom', fontsize=4.9, color=INK2)
    ax = fig.add_subplot(gs[0, 0])
    fam = ['Primesep (mixed-mode)', 'Obelisc (mixed-mode)', 'Newcrom (RP / mixed-mode)', 'Legacy L1 (C18)', 'Cannsep', 'SHARC']
    grouped(ax, 'family', fam, ['Primesep', 'Obelisc', 'Newcrom', 'Legacy L1 (C18)', 'Cannsep', 'SHARC'], 'Stationary phase',
            f"χ²({S['family_dof']}) = {S['family_chi2']:.1f}, P < 10$^{{-12}}$, V = {S['family_V']:.2f}")
    ax.legend(loc='lower right'); label(ax, 'a', x=-0.42)
    ax = fig.add_subplot(gs[0, 1])
    add = ['Phosphate / H3PO4', 'Ammonium acetate', 'Ammonium formate', 'H2SO4', 'TFA', 'Formic acid', 'Other', 'No additive']
    grouped(ax, 'additive', add, ['Phosphate/H$_3$PO$_4$', 'NH$_4$ acetate', 'NH$_4$ formate', 'H$_2$SO$_4$', 'TFA', 'Formic acid', 'Other', 'None'], 'Additive',
            f"χ²({S['additive_dof']}) = {S['additive_chi2']:.1f}, P = {S['additive_p']:.0e}, V = {S['additive_V']:.2f}")
    label(ax, 'b', x=-0.42)
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(5)
    for i, c in enumerate(cls):
        v = kb[kb['class'] == c].wl.dropna().values
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.5, color=INK2),
                   capprops=dict(lw=0.5, color=INK2), boxprops=dict(lw=0.5, color=INK2))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v + rng.uniform(-1, 1, len(v)), s=5, color=col[c], alpha=0.75, lw=0.25, ec='white', zorder=3)
    ax.set_xticks([0, 1], cls); ax.set_ylabel('Detection wavelength (nm)'); ax.set_title('UV detection', pad=10)
    ax.text(1.0, 1.01, f"Mann–Whitney P = {S['wl_mw_p']:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=4.9, color=INK2); label(ax, 'c', x=-0.3)
    ax = fig.add_subplot(gs[1, 0])
    for i, c in enumerate(cls):
        v = st[st['class'] == c].logP.values
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.5, color=INK2),
                   capprops=dict(lw=0.5, color=INK2), boxprops=dict(lw=0.5, color=INK2))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color=col[c], alpha=0.8, lw=0.25, ec='white', zorder=3)
    ax.set_xticks([0, 1], [f'{c}\n(n = {(st["class"]==c).sum()})' for c in cls]); ax.set_ylabel('Crippen logP')
    ax.set_title('Structure vs class label', pad=10); ax.text(1.0, 1.01, f"Mann–Whitney P = {ss['logP']['p']:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=4.9, color=INK2)
    label(ax, 'd', x=-0.42)
    ax = fig.add_subplot(gs[1, 1])
    yc = (st['class'] == 'Hydrophilic').astype(int); yf = (st.family == 'Primesep (mixed-mode)').astype(int)
    for yv, p, c, lab in ((yc, st.p_class_rf, ORANGE, f"Class (AUC = {ss['class_auc_rf']:.2f})"), (yf, st.p_primesep_rf, BLUE, f"Primesep column (AUC = {ss['family_auc_rf']:.2f})")):
        fpr, tpr, _ = roc_curve(yv, p); ax.plot(fpr, tpr, color=c, lw=1, label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.5, ls=(0, (2, 1.5))); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.set_aspect('equal')
    ax.legend(loc='lower right'); ax.set_title('Structure → condition (RF, 5-fold CV)', pad=10); label(ax, 'e', x=-0.42)
    ax = fig.add_subplot(gs[1, 2])
    for i, c in enumerate(cls):
        sub = st[st['class'] == c]
        ax.scatter(sub.logP, sub.TPSA, s=7, color=col[c], alpha=0.8, ec='white', lw=0.25, label=c)
    ax.set_xlabel('Crippen logP'); ax.set_ylabel('TPSA (Å$^2$)'); ax.set_ylim(-3, 92); ax.legend(loc='upper center', ncol=2, columnspacing=0.6, handletextpad=0.2); ax.set_title(f"Structures assigned (n = {ss['n_struct']})", pad=10)
    label(ax, 'f', x=-0.3)
    save(fig, 'Fig3_condition_knowledge_base')


MOD_ORDER = ['Ridge', 'RF', 'LightGBM', 'ANN']; SCH = {'compound': ('New compound', BLUE), 'dataset': ('New system', ORANGE), 'drug-external': ('New drug', AQUA)}
def hexpar(ax, yt, yp, title):
    ax.hexbin(yt, yp, gridsize=55, cmap=cmap_seq, bins='log', mincnt=1, lw=0, extent=(0, 100, 0, 100))
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.5, ls=(0, (2, 1.5)))
    for d in (-10, 10): ax.plot([0, 100], [d, 100 + d], color=MUTED, lw=0.4, ls=':')
    mae = np.mean(np.abs(yt - yp)); r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    ax.text(0.04, 0.96, f'MAE = {mae:.1f}%\nR² = {r2:.2f}\nn = {len(yt):,}', transform=ax.transAxes, va='top', fontsize=5.3, color=INK2)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal'); ax.set_xlabel('Literature φ$_e$ (%)'); ax.set_ylabel('Predicted φ$_e$ (%)'); ax.set_title(title, pad=4)

def figBench():
    R = pd.read_csv(D + 'kb2_results.csv'); P = pd.read_csv(D + 'kb2_predictions.csv'); LC = pd.read_csv(D + 'kb2_learning_curve.csv')
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.45, hspace=0.55, left=0.08, right=0.985, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0]); y = np.arange(len(MOD_ORDER))
    for k, (sch, (lab, c)) in enumerate(SCH.items()):
        sub = R[R.scheme == sch].set_index('model').reindex(MOD_ORDER)
        ax.scatter(sub.MAE, y + (k - 1) * 0.22, color=c, s=14, ec='white', lw=0.4, label=lab, zorder=3)
        for yy, v in zip(y + (k - 1) * 0.22, sub.MAE): ax.hlines(yy, 0, v, color=GRID, lw=0.6, zorder=1)
    base = R[(R.model == 'Dataset mean')].MAE.iloc[0]
    ax.axvline(base, color=MUTED, lw=0.5, ls=(0, (2, 1.5))); ax.text(base - 0.3, 3.35, 'System-mean\nbaseline', fontsize=4.8, color=MUTED, ha='right', va='center')
    ax.set_yticks(y, MOD_ORDER); ax.set_ylim(len(MOD_ORDER) - 0.4, -0.9); ax.set_xlim(0, max(base, R.MAE.max()) * 1.08); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('MAE of φ$_e$ (% organic)'); ax.legend(loc='lower right'); ax.set_title('Model benchmark', pad=4); label(ax, 'a', x=-0.25)
    best = R[R.scheme == 'compound'].set_index('model').reindex(MOD_ORDER).MAE.idxmin()
    ax = fig.add_subplot(gs[0, 1]); hexpar(ax, P.phi_e.values, P[f'oof_compound_{best}'].values, f'{best}: new compound (5-fold)'); label(ax, 'b', x=-0.25)
    ax = fig.add_subplot(gs[0, 2]); hexpar(ax, P.phi_e.values, P[f'oof_dataset_{best}'].values, f'{best}: new system (5-fold)'); label(ax, 'c', x=-0.25)
    ax = fig.add_subplot(gs[1, 0])
    for m, c in (('LightGBM', BLUE), ('ANN', ORANGE)):
        g = LC[LC.model == m].groupby('n').MAE.agg(['mean', 'std']).reset_index()
        ax.errorbar(g.n, g['mean'], yerr=g['std'].fillna(0), color=c, marker='o', ms=3, lw=0.9, capsize=1.5, elinewidth=0.6, label=m)
    ax.set_xscale('log'); ax.set_xlabel('Training records (n)'); ax.set_ylabel('MAE of φ$_e$ (%)'); ax.legend(); ax.set_title('Learning curve', pad=4); label(ax, 'd', x=-0.25)
    ax = fig.add_subplot(gs[1, 1])
    labs = ['Dataset mean', 'LightGBM System only', 'LightGBM Molecular only', 'LightGBM']
    vals = [R[(R.scheme == 'compound') & (R.model == l)].MAE.iloc[0] for l in labs]
    names = ['System mean', 'System features', 'Molecular features', 'Both']
    ax.barh(range(4), vals, color=[GRID, SEQ[3], SEQ[5], BLUE], height=0.6, lw=0)
    for i, v in enumerate(vals): ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=5, color=INK2)
    ax.set_yticks(range(4), names); ax.invert_yaxis(); ax.tick_params(axis='y', length=0); ax.set_xlabel('MAE of φ$_e$ (%)'); ax.set_title('Feature ablation (LightGBM)', pad=4); label(ax, 'e', x=-0.42)
    ax = fig.add_subplot(gs[1, 2])
    P['e'] = (P[f'oof_dataset_{best}'] - P.phi_e).abs(); per = P.groupby('dataset').e.mean()
    ax.hist(per, bins=30, color=ORANGE, lw=0); ax.axvline(per.median(), color=INK, lw=0.7)
    ax.text(per.median() + 0.5, ax.get_ylim()[1] * 0.92, f'median {per.median():.1f}%', fontsize=5, color=INK2)
    ax.set_xlabel('Per-system MAE, new system (%)'); ax.set_ylabel('Systems (n)'); ax.set_title(f'Transfer to unseen systems (n = {len(per)})', pad=4); label(ax, 'f', x=-0.25)
    save(fig, 'Fig4_model_benchmark')

def figInterp():
    mods = pickle.load(open(D + 'kb2_models.pkl', 'rb')); FEATS = mods['FEATS']; molset = set(mods['molcols'])
    sv = np.load(D + 'kb2_shap.npy'); idx = np.load(D + 'kb2_shap_idx.npy'); P = pd.read_csv(D + 'kb2_predictions.csv')
    imp = np.abs(sv).mean(0); o = np.argsort(imp)[::-1][:15]
    NICE = {'MolLogP': 'logP (Crippen)', 'phi0': 'Initial φ', 'b': 'Gradient steepness b', 'slope': 'Gradient slope', 't0': 'Dead time t$_0$', 'tG': 'Gradient time',
            'phimax': 'Final φ', 'acn_frac': 'ACN fraction', 'TPSA': 'TPSA', 'MolWt': 'Mol. weight', 'flow': 'Flow rate', 'L': 'Column length', 'T': 'Temperature',
            'add_formic': 'Formic acid', 'add_nh4ac': 'NH$_4$ acetate', 'add_nh4form': 'NH$_4$ formate', 'pHA': 'pH (eluent A)', 'H': 'HSM H', 'kPB': 'Tanaka k$_{PB}$',
            'C28': 'HSM C(2.8)', 'ID': 'Column i.d.', 'dp': 'Particle size'}
    fig = plt.figure(figsize=(W2, 66 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.13, right=0.985, top=0.88, bottom=0.18, width_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(gs[0])
    for i, j in enumerate(o):
        ax.barh(i, imp[j], color=BLUE if FEATS[j] in molset else ORANGE, height=0.65, lw=0)
    ax.set_yticks(range(len(o)), [NICE.get(FEATS[j], FEATS[j]) for j in o]); ax.invert_yaxis(); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Mean |SHAP| (% organic)')
    molshare = imp[[i for i, f in enumerate(FEATS) if f in molset]].sum() / imp.sum()
    ax.legend(handles=[Patch(fc=BLUE, label=f'Molecular ({100*molshare:.0f}% of total)'), Patch(fc=ORANGE, label=f'System ({100*(1-molshare):.0f}%)')], loc='lower right')
    ax.set_title('Feature importance (LightGBM, SHAP)', pad=4); label(ax, 'a', x=-0.5)
    ax = fig.add_subplot(gs[1]); j = FEATS.index('MolLogP')
    X = pd.read_pickle(D + 'kb2_desc.pkl').set_index('inchikey')
    lp = P.iloc[idx].ik14.map(X.MolLogP.groupby(X.index.str[:14]).first()).values
    sc_ = ax.scatter(lp, sv[:, j], s=3, c=P.iloc[idx].phi_e.values, cmap=cmap_seq, lw=0, alpha=0.8, rasterized=True)
    ax.axhline(0, color=MUTED, lw=0.4); ax.set_xlabel('logP (Crippen)'); ax.set_ylabel('SHAP value for logP (%)'); ax.set_xlim(-6, 12)
    cb = fig.colorbar(sc_, ax=ax, fraction=0.05, pad=0.02); cb.outline.set_visible(False); cb.set_label('φ$_e$ (%)', fontsize=5); cb.ax.tick_params(labelsize=5)
    ax.set_title('Hydrophobicity dependence', pad=4); label(ax, 'b', x=-0.3)
    ax = fig.add_subplot(gs[2])
    best = 'LightGBM'
    P['err'] = (P[f'oof_compound_{best}'] - P.phi_e).abs(); P['dec'] = pd.qcut(P.ad_dist, 10, labels=False)
    g = P.groupby('dec').agg(med=('err', 'median'), q1=('err', lambda v: v.quantile(.25)), q3=('err', lambda v: v.quantile(.75)), d=('ad_dist', 'median'))
    ax.fill_between(g.index + 1, g.q1, g.q3, color=BLUE, alpha=0.18, lw=0); ax.plot(g.index + 1, g.med, color=BLUE, marker='o', ms=3, lw=0.9)
    ax.set_xlabel('Distance to training set (decile)'); ax.set_ylabel('Absolute error (%)'); ax.set_xticks(range(1, 11))
    from scipy.stats import spearmanr
    rho = spearmanr(P.ad_dist, P.err, nan_policy='omit')
    ax.text(0.04, 0.96, f'Spearman ρ = {rho.correlation:.2f}\nP < 10$^{{-10}}$' if rho.pvalue < 1e-10 else f'ρ = {rho.correlation:.2f}, P = {rho.pvalue:.2g}', transform=ax.transAxes, va='top', fontsize=5.2, color=INK2)
    ax.set_title('Applicability domain', pad=4); label(ax, 'c', x=-0.3)
    save(fig, 'Fig5_interpretability')
    json.dump(dict(molshare=float(molshare), top=[FEATS[j] for j in o], rho=float(rho.correlation), dec1=float(g.med.iloc[0]), dec10=float(g.med.iloc[-1])), open(D + 'interp.json', 'w'))

def figApp():
    E = pd.read_csv(D + 'drug_external.csv'); L = pd.read_csv(D + 'case_literature_check.csv'); Sc = pd.read_csv(D + 'scouting.csv', dtype={'dataset': str})
    Pr = pd.read_csv(D + 'scouting_profiles.csv', dtype={'dataset': str}); R = pd.read_csv(D + 'kb2_results.csv')
    fig = plt.figure(figsize=(W2, 125 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.45, left=0.09, right=0.985, top=0.94, bottom=0.07, width_ratios=[1, 1.25])
    gs = fig.add_gridspec(2, 2, wspace=0.32, left=0.09, right=0.985, top=0.94, bottom=0.55, width_ratios=[1, 1.25], height_ratios=[1, 0.001], hspace=0)
    ax = fig.add_subplot(gs[0, 0])
    order = ['cardiovascular', 'cns', 'antibiotic', 'analgesic', 'antidiabetic', 'other', 'case study']
    order = [o for o in order if o in E.cls.unique()]
    rng = np.random.default_rng(2)
    for i, c in enumerate(order):
        for k, (col_, cc) in enumerate((('err_lgb', BLUE), ('err_ann', ORANGE))):
            v = E[E.cls == c][col_].values; pos = i + (k - 0.5) * 0.36
            ax.boxplot(v, positions=[pos], widths=0.3, vert=False, showfliers=False, medianprops=dict(color=cc, lw=1), whiskerprops=dict(lw=0.5, color=INK2),
                       capprops=dict(lw=0.5, color=INK2), boxprops=dict(lw=0.5, color=INK2))
        ax.text(47, i, f"n = {(E.cls == c).sum()}", va='center', ha='right', fontsize=4.8, color=MUTED)
    ax.set_yticks(range(len(order)), [CLS_LAB.get(c, 'Case-study drugs') for c in order]); ax.invert_yaxis(); ax.set_xlim(0, 48); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Absolute error of φ$_e$ (%), drugs held out')
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1, label='LightGBM'), Line2D([], [], color=ORANGE, lw=1, label='ANN')], loc='lower left', bbox_to_anchor=(0.45, 1.0), ncol=2)
    ax.set_title('External drug test by class', pad=4); label(ax, 'a', x=-0.3)
    ax = fig.add_subplot(gs[0, 1])
    drugs = L.drug.unique(); mk_ = ['o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', 'h']
    mx = np.nanmax([L.rt_lit.max(), L.rt_pred.max()]) * 1.05
    ax.plot([0, mx], [0, mx], color=INK2, lw=0.5, ls=(0, (2, 1.5)))
    for k, d in enumerate(drugs):
        s = L[L.drug == d]; c = BLUE if s.panel.iloc[0] == 'A' else ORANGE
        ax.scatter(s.rt_lit, s.rt_pred, marker=mk_[k % len(mk_)], s=12, facecolor='white', ec=c, lw=0.7, label=f'{d} ({len(s)})', zorder=3)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_xlabel('Published t$_R$ (min)'); ax.set_ylabel('Predicted t$_R$ (min)'); ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5, handletextpad=0.3)
    ax.text(0.97, 0.04, f"MAE = {L.abs_err_rt.mean():.2f} min\nmedian = {L.abs_err_rt.median():.2f} min\nn = {len(L)} in {L.dataset.nunique()} systems", transform=ax.transAxes, ha='right', va='bottom', fontsize=5.2, color=INK2)
    ax.set_title('Case-study drugs vs published methods', pad=4); label(ax, 'b', x=-0.25)
    mae_sys = R[(R.scheme == 'compound') & (R.model == 'LightGBM')].MAE.iloc[0]
    MK = ['o', 's', '^', 'v', 'D', 'P', 'X', 'h']
    for k, panel in enumerate(('A', 'B')):
        ax = fig.add_axes([0.2, 0.08, 0.32, 0.33] if k == 0 else [0.71, 0.08, 0.27, 0.33])
        top = Sc[Sc.panel == panel].sort_values('score', ascending=False).head(4)
        c = BLUE if panel == 'A' else ORANGE
        dlist = sorted(Pr[Pr.panel == panel].drug.unique(), key=lambda d: d)
        for i, (_, s) in enumerate(top.iterrows()):
            pr = Pr[(Pr.panel == panel) & (Pr.dataset == s.dataset)]
            ax.hlines(i, 0, s.run, color=GRID, lw=0.6)
            for _, p in pr.iterrows():
                ax.scatter(p.rt, i, marker=MK[dlist.index(p.drug)], s=18, facecolor='white', ec=c, lw=0.8, zorder=3)
            nm = s.column
            for pre in ('Waters ', 'Agilent ', 'Phenomenex ', 'Thermo Scientific ', 'Merck Supelco '): nm = nm.replace(pre, '')
            ax.text(-0.02, i, f"{nm[:24]}\n{'ACN' if s.acn_frac >= .5 else 'MeOH'}; min gap {s.min_gap:.2f} min", transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=4.8, color=INK)
        ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_ylim(len(top) - 0.5, -0.6)
        ax.set_xlim(0, top.run.max() * 1.02); ax.set_xlabel('Predicted retention time (min)')
        ax.legend(handles=[Line2D([], [], marker=MK[j], ls='', mfc='white', mec=c, ms=4, label=d.replace('Isosorbide mononitrate', 'ISMN')) for j, d in enumerate(dlist)],
                  loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4 if panel == 'A' else 3, handletextpad=0.2, columnspacing=0.8, fontsize=5)
        ax.text(-0.02, 1.16 if panel == 'A' else 1.1, f"Panel {panel}: top 4 of {len(Sc[Sc.panel==panel])} C18 systems", transform=ax.transAxes, ha='right', fontsize=6, color=INK)
        ax.text(-0.62 if k == 0 else -0.52, 1.2, 'cd'[k], transform=ax.transAxes, fontsize=8, fontweight='bold')
    save(fig, 'Fig6_drug_application')

if __name__ == '__main__':
    which = sys.argv[1:] or ['W', 'KB', 'SIELC']
    for w in which: globals()['fig' + w](); print('ok', w)
