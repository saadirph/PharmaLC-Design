import sys, json, math
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
sys.path.insert(0, '/home/claude/scripts')
from data import *

MM = 1 / 25.4
W2, W1 = 183 * MM, 89 * MM
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#1a1a1a', '#52514e', '#8a8984', '#e6e5e1'
SEQ = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
cmap_seq = LinearSegmentedColormap.from_list('seq', SEQ)

mpl.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 6, 'axes.titlesize': 6.5, 'axes.labelsize': 6,
    'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 5.5,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left', 'lines.linewidth': 0.8,
})
OUTD = '/home/claude/figs/'

def label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')

def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUTD + f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUTD + f'{name}.tiff', bbox_inches='tight', pad_inches=0.02, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

SHORT = {"Isosorbide mononitrate": "Isosorbide mononitrate"}
def panel_color(d): return BLUE if d in NINE else ORANGE

# ============ Figure 1: workflow schematic ============
def fig1():
    fig = plt.figure(figsize=(W2, 46 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 46); ax.axis('off')
    def box(x, y, w, h, title, body, fc='#ffffff', ec=INK2, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fc, ec=ec, lw=0.6))
        ax.text(x + 2.2, y + h - 2.6, title, fontsize=6.5, fontweight='bold', va='top', color=tc)
        ax.text(x + 2.2, y + h - 7.4, body, fontsize=5.4, va='top', color=INK2, linespacing=1.35)
    def arrow(x0, y0, x1, y1):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK2, mutation_scale=6, shrinkA=0, shrinkB=0))
    box(2, 16, 38, 28, 'Knowledge base', '164 unique compounds\n83 hydrophobic · 81 hydrophilic\nColumn, mobile phase,\nadditive, flow rate,\ndetection wavelength\n(SIELC application data)', fc='#f4f8fd', ec=BLUE)
    box(48, 16, 38, 28, 'ANN-MLP model', 'Inputs (covariates):\nstructure, column, mobile\nphase, wavelength, flow rate\nHidden: tanh · Output: identity\nLoss: sum of squares\nStop: 1 step w/o improvement')
    box(94, 16, 38, 28, 'Experimental method', 'C18 ODS, 250 × 4.6 mm\nMeOH–ACN 50:50 (v/v)\nIsocratic, UV detection\nPlasma: protein removal\n[authors to specify]')
    box(140, 27, 41, 17, 'Panel A · 9 drugs', 'Cardiovascular / metabolic:\nAML, BIS, CLA, CLO, FA,\nGLM, ISMN, MCB, PCM', fc='#f4f8fd', ec=BLUE)
    box(140, 2, 41, 17, 'Panel B · 3 CNS drugs', 'Alprazolam, bromazepam,\nmethylphenidate', fc='#fdf1ec', ec=ORANGE)
    box(48, 1, 84, 11, 'Validation (ICH Q2(R2) / ICH M10)', 'SST · selectivity · linearity · accuracy · precision · LOD/LOQ · robustness · ruggedness · stability')
    arrow(40, 30, 48, 30); arrow(86, 30, 94, 30); arrow(132, 34, 140, 36); arrow(132, 22, 140, 12)
    arrow(113, 16, 113, 12)
    save(fig, 'Fig1_workflow')

# ============ Figure 2: knowledge base ============
def fig2():
    kb = pd.read_csv('/home/claude/data/knowledge_base_clean.csv')
    S = json.load(open('/home/claude/data/summary.json'))['kb_stats']
    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.55, left=0.17, right=0.98, top=0.95, bottom=0.08)
    cls = ['Hydrophobic', 'Hydrophilic']; col = {'Hydrophobic': BLUE, 'Hydrophilic': ORANGE}
    def grouped(ax, var, order, title, stat):
        ct = pd.crosstab(kb[var], kb['class']).reindex(order).fillna(0)
        y = np.arange(len(order)); h = 0.36
        for i, c in enumerate(cls):
            ax.barh(y + (i - 0.5) * (h + 0.04), ct[c], height=h, color=col[c], label=f'{c} (n = {int((kb["class"]==c).sum())})', lw=0)
            for yy, v in zip(y + (i - 0.5) * (h + 0.04), ct[c]):
                if v > 0: ax.text(v + 0.8, yy, f'{int(v)}', va='center', fontsize=5, color=INK2)
        ax.set_yticks(y, order); ax.invert_yaxis(); ax.set_xlabel('Compounds (n)')
        ax.tick_params(axis='y', length=0); ax.set_title(title, pad=10)
        ax.text(1.0, 1.01, stat, transform=ax.transAxes, ha='right', va='bottom', fontsize=5.3, color=INK2)
        ax.xaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[0, 0])
    fam_order = ['Primesep (mixed-mode)', 'Obelisc (mixed-mode)', 'Newcrom (RP / mixed-mode)', 'Legacy L1 (C18)', 'Cannsep', 'SHARC']
    grouped(ax, 'family', fam_order, 'Stationary-phase family',
            f"χ²({S['family_dof']}) = {S['family_chi2']:.1f}, P = {S['family_p']:.1e}; Cramér's V = {S['family_V']:.2f}")
    ax.legend(loc='lower right', handlelength=1, handleheight=0.7)
    label(ax, 'a', x=-0.28)
    ax = fig.add_subplot(gs[0, 1])
    add_order = ['Phosphate / H3PO4', 'Ammonium acetate', 'Ammonium formate', 'H2SO4', 'TFA', 'Formic acid', 'Other', 'No additive']
    grouped(ax, 'additive', add_order, 'Mobile-phase additive',
            f"χ²({S['additive_dof']}) = {S['additive_chi2']:.1f}, P = {S['additive_p']:.1e}; V = {S['additive_V']:.2f}")
    ax.set_yticklabels(['Phosphate / H$_3$PO$_4$', 'Ammonium acetate', 'Ammonium formate', 'H$_2$SO$_4$', 'TFA', 'Formic acid', 'Other', 'No additive'])
    label(ax, 'b', x=-0.28)
    # c wavelength
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(7)
    for i, c in enumerate(cls):
        v = kb[kb['class'] == c].wl.dropna().values
        bp = ax.boxplot(v, positions=[i], widths=0.42, vert=False, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.5, color=INK2), capprops=dict(lw=0.5, color=INK2),
                        boxprops=dict(lw=0.5, ec=INK2, fc='none'))
        ax.scatter(v + rng.uniform(-1.2, 1.2, len(v)), i + rng.uniform(-0.17, 0.17, len(v)), s=7, color=col[c], alpha=0.75, lw=0.3, ec='white', zorder=3)
    ax.set_yticks([0, 1], [f'{c}\n(n = {int(kb[kb["class"]==c].wl.notna().sum())})' for c in cls]); ax.invert_yaxis()
    ax.set_xlabel('UV detection wavelength (nm)'); ax.set_title('Detection wavelength', pad=10); ax.tick_params(axis='y', length=0)
    ax.text(1.0, 1.01, f"Mann–Whitney U = {S['wl_mw_U']:.0f}, P = {S['wl_mw_p']:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=5.3, color=INK2)
    ax.set_xlim(190, 290); ax.xaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    label(ax, 'c', x=-0.28)
    # d flow & elution mode
    ax = fig.add_subplot(gs[1, 1])
    flows = [0.4, 0.5, 0.6, 1.0, 1.2, 1.5]
    x = np.arange(len(flows)); w = 0.36
    for i, c in enumerate(cls):
        sub = kb[kb['class'] == c]
        iso = [((sub.flow_num == f) & ~sub.gradient).sum() for f in flows]
        gra = [((sub.flow_num == f) & sub.gradient).sum() for f in flows]
        xx = x + (i - 0.5) * (w + 0.04)
        iso = np.array(iso); gra = np.array(gra)
        m = iso > 0; ax.bar(xx[m], iso[m], width=w, color=col[c], lw=0)
        m = gra > 0; ax.bar(xx[m], gra[m], width=w, bottom=iso[m] + np.where(iso[m] > 0, 0.6, 0), color='white', ec=col[c], lw=0.6, hatch='//////')
    mpl.rcParams['hatch.linewidth'] = 0.4
    ax.set_xticks(x, [f'{f:g}' for f in flows]); ax.set_xlabel('Flow rate (mL min$^{-1}$)'); ax.set_ylabel('Compounds (n)')
    ax.set_title('Flow rate and elution mode', pad=10)
    g = S['grad_table']
    ax.text(1.0, 1.01, f"Gradient: {g[1][1]}/{g[0][1]+g[1][1]} vs {g[1][0]}/{g[0][0]+g[1][0]}; Fisher P = {S['fisher_p']:.2f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.3, color=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=BLUE, lw=0, label='Hydrophobic'), Patch(fc=ORANGE, lw=0, label='Hydrophilic'),
                       Patch(fc='white', ec=INK2, hatch='//////', lw=0.6, label='Gradient (hatched)')], loc='upper left')
    ax.yaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    label(ax, 'd', x=-0.12)
    save(fig, 'Fig2_knowledge_base')

# ============ Figure 3: system suitability ============
def fig3():
    order = NINE + CNS
    fig, axs = plt.subplots(1, 4, figsize=(W2, 62 * MM), sharey=True, gridspec_kw=dict(wspace=0.22, left=0.15, right=0.99, top=0.86, bottom=0.17))
    y = np.arange(len(order))
    tr = [SST[d][0] for d in order]; rs = [SST[d][1] for d in order]; N = [SST[d][2] for d in order]; tf = [SST[d][3] for d in order]
    N = [6781.6 if d == 'Paracetamol' else n for d, n in zip(order, N)]
    cols = [panel_color(d) for d in order]
    specs = [(tr, 'Retention time (min)', None, None, (0, 10)),
             (N, 'Theoretical plates, N', 2000, 'N > 2,000', (0, 9000)),
             (tf, 'Tailing factor, T', 2.0, 'T ≤ 2.0', (0.8, 2.8)),
             (rs, 'Resolution, R$_s$', 2.0, 'R$_s$ ≥ 2.0', (0, 10))]
    for k, (ax, (v, xl, crit, ct, lim)) in enumerate(zip(axs, specs)):
        for yi, val, c, d in zip(y, v, cols, order):
            if val is None:
                ax.text(lim[0] + 0.2, yi, 'n.r.', va='center', fontsize=5, color=MUTED); continue
            fail = crit is not None and ((k == 2 and val > crit) or (k in (1, 3) and val < crit))
            ax.hlines(yi, lim[0], val, color=GRID, lw=0.6, zorder=1)
            ax.scatter(val, yi, s=16, color='white' if fail else c, ec=c, lw=0.9, zorder=3)
            if (k == 1 and d == 'Paracetamol'):
                ax.text(val + 200, yi, '*', va='center', fontsize=7, color=INK)
        if crit is not None:
            if k == 2: ax.axvspan(crit, lim[1], color='#f6e3dc', lw=0, zorder=0)
            else: ax.axvspan(lim[0], crit, color='#f6e3dc', lw=0, zorder=0)
            ax.axvline(crit, color=INK2, lw=0.5, ls=(0, (2, 1.5)))
            ax.text(crit + (0.02 if k == 2 else 0) * 0, -0.75, ' ' + ct + ' ', fontsize=5, color=INK2, ha='left' if k != 2 else 'right', va='bottom')
        ax.set_xlim(*lim); ax.set_xlabel(xl); ax.set_ylim(len(order) - 0.4, -1.3)
        ax.tick_params(axis='y', length=0)
        label(ax, 'abcd'[k], x=-0.04 if k else -0.5)
    axs[0].set_yticks(y, order)
    axs[0].axhline(8.5, color=MUTED, lw=0.4, ls=':')
    for ax in axs[1:]: ax.axhline(8.5, color=MUTED, lw=0.4, ls=':')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, ms=4, label='Panel A (9 drugs)'),
                        Line2D([], [], marker='o', ls='', color=ORANGE, ms=4, label='Panel B (CNS)'),
                        Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=4, label='Outside acceptance limit')],
               loc='upper center', ncol=3, bbox_to_anchor=(0.57, 1.0))
    save(fig, 'Fig3_system_suitability')

# ============ Figure 4: accuracy ============
def fig4():
    acc = pd.read_csv('/home/claude/data/accuracy.csv')
    acc['levels'] = acc.levels.apply(eval)
    S = json.load(open('/home/claude/data/summary.json'))
    fig = plt.figure(figsize=(W2, 72 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.32, left=0.07, right=0.99, top=0.88, bottom=0.27)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(NINE))
    ax.axhspan(98, 102, color='#eef4fc', lw=0, zorder=0)
    ax.axhline(100, color=INK2, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    marks = {0: ('v', '75 %'), 1: ('o', '100 %'), 2: ('^', '125 %')}
    for i, r in acc.iterrows():
        for j, v in enumerate(r.levels):
            ax.scatter(i + (j - 1) * 0.14, v, marker=marks[j][0], s=11, color='white', ec=BLUE, lw=0.7, zorder=3)
        ax.errorbar(i + 0.32, r['mean'], yerr=r.ci95, fmt='s', ms=3.2, color=BLUE, mec=BLUE, elinewidth=0.8, capsize=1.6, capthick=0.6, zorder=4)
    ax.set_xticks(x, NINE, rotation=40, ha='right', rotation_mode='anchor'); ax.set_xlim(-0.6, len(NINE) - 0.3)
    ax.set_ylabel('Recovery (%)'); ax.set_ylim(94.5, 104.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, ls='', mfc='white', mec=BLUE, ms=3.6, label=f'Spike {l}') for m, l in marks.values()]
    h.append(Line2D([], [], marker='s', color=BLUE, ms=3, lw=0.8, label='Mean ± 95% CI'))
    ax.legend(handles=h, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.13))
    p = S['acc_pooled']
    ax.text(0.99, 0.04, f"Pooled (n = {p['n']}): {p['mean']:.2f} ± {p['ci']:.2f}%\nFriedman (spike level): P = {S['acc_level_friedman']['pvalue']:.2f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.3, color=INK2)
    ax.text(len(NINE) - 0.35, 102.1, '98–102%', fontsize=5, color=MUTED, ha='right', va='bottom')
    label(ax, 'a', x=-0.06, y=1.08)
    # b CNS
    ax = fig.add_subplot(gs[1])
    xs = np.repeat([25, 50, 75], 3).astype(float); ys = np.concatenate([ACC_CNS[k] for k in (25, 50, 75)])
    res = stats.linregress(xs, ys)
    xx = np.linspace(20, 80, 100); n = len(xs)
    yhat = res.intercept + res.slope * xx
    s = np.sqrt(((ys - (res.intercept + res.slope * xs)) ** 2).sum() / (n - 2))
    se = s * np.sqrt(1 / n + (xx - xs.mean()) ** 2 / ((xs - xs.mean()) ** 2).sum())
    tq = stats.t.ppf(0.975, n - 2)
    ax.axhspan(98, 102, color='#fdf1ec', lw=0, zorder=0); ax.axhline(100, color=INK2, lw=0.5, ls=(0, (2, 1.5)))
    ax.fill_between(xx, yhat - tq * se, yhat + tq * se, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(xx, yhat, color=ORANGE, lw=1)
    rng = np.random.default_rng(3)
    ax.scatter(xs + rng.uniform(-1.2, 1.2, n), ys, s=12, color=ORANGE, ec='white', lw=0.4, zorder=3)
    c = S['acc_cns']
    ax.set_xticks([25, 50, 75]); ax.set_xlim(17, 83); ax.set_ylim(97, 103)
    ax.set_xlabel('Standard added to 50 µg mL$^{-1}$ sample (µg mL$^{-1}$)'); ax.set_ylabel('Recovery (%)')
    ax.text(0.03, 0.97, f"Slope = {res.slope*10:.2f}% per 10 µg mL$^{{-1}}$\nP = {res.pvalue:.3f}, R² = {res.rvalue**2:.2f}\nANOVA F(2,6) = {c['anova_F']:.1f}, P = {c['anova_p']:.3f}",
            transform=ax.transAxes, va='top', fontsize=5.3, color=INK2)
    ax.set_title('Panel B (CNS)', pad=4); fig.axes[0].set_title('Panel A (nine drugs), standard addition', pad=14)
    label(ax, 'b', x=-0.16, y=1.08)
    save(fig, 'Fig4_accuracy')

# ============ Figure 5: precision, robustness, ruggedness ============
def fig5():
    inter = pd.read_csv('/home/claude/data/interday_anova.csv')
    cols = ['Intraday', 'Day 1', 'Day 2', 'Day 3', 'Flow +0.2', 'Flow −0.2', 'Temp +2 °C', 'Temp −2 °C', 'MP +2%', 'MP −2%', 'Analyst A', 'Analyst B']
    M = []; Q = []
    for d in NINE:
        row = [INTRA[d][1]] + [r for _, r in INTER[d]] + ROB[d] + list(RUG[d])
        q = [v >= 10 for v in row]
        M.append([np.nan if qq else v for v, qq in zip(row, q)]); Q.append(q)
    M = np.array(M, float); Q = np.array(Q)
    fig = plt.figure(figsize=(W2, 80 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 0.7], wspace=0.12, left=0.13, right=0.985, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M, cmap=cmap_seq, vmin=0, vmax=1.5, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if Q[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#ecebe7', ec='white', lw=0.8, hatch='////'))
                ax.text(j, i, '†', ha='center', va='center', fontsize=6, color=INK)
            else:
                v = M[i, j]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4.6, color='white' if v > 0.95 else INK)
    for xg in np.arange(-0.5, M.shape[1], 1): ax.axvline(xg, color='white', lw=0.8)
    for yg in np.arange(-0.5, M.shape[0], 1): ax.axhline(yg, color='white', lw=0.8)
    for xb in (0.5, 3.5, 9.5): ax.axvline(xb, color=INK, lw=0.6)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha='right', rotation_mode='anchor'); ax.set_yticks(range(len(NINE)), NINE)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for x0, x1, t in ((0, 0, 'Repeat.'), (1, 3, 'Interday'), (4, 9, 'Robustness'), (10, 11, 'Ruggedness')):
        ax.text((x0 + x1) / 2, -0.85, t, ha='center', va='bottom', fontsize=5.5, color=INK2)
    cax = fig.add_axes([0.13, 0.955, 0.16, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5, length=1.5)
    cax.set_title('%RSD', fontsize=5.5, loc='left', pad=2); cb.set_ticks([0, 0.5, 1.0, 1.5])
    label(ax, 'a', x=-0.3, y=1.06)
    # b interday means ± 95% CI
    ax = fig.add_subplot(gs[1], sharey=ax)
    dayc = ['#9ec5f4', '#3987e5', '#184f95']
    for i, d in enumerate(NINE):
        for k, (m, r) in enumerate(INTER[d]):
            rr = r / 10 if r >= 10 else r
            ci = stats.t.ppf(0.975, 5) * m * rr / 100 / math.sqrt(6)
            yy = i + (k - 1) * 0.24
            ax.errorbar(m, yy, xerr=ci, fmt='o', ms=2.6, color=dayc[k], elinewidth=0.7, capsize=0, mfc=dayc[k] if r < 10 else 'white')
        pr = inter.set_index('drug').loc[d]
        ax.text(101.2, i, f"P = {pr.p:.2f}" + ('†' if pr.flagged else ''), va='center', fontsize=5, color=INK2)
    ax.axvline(100, color=INK2, lw=0.5, ls=(0, (2, 1.5)))
    ax.set_xlim(97.8, 102.2); ax.set_xticks([98, 99, 100, 101])
    ax.set_xlabel('Assay (%), mean ± 95% CI (n = 6)'); ax.tick_params(axis='y', labelleft=False, length=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', color=c, ms=3, lw=0.7, label=f'Day {k+1}') for k, c in enumerate(dayc)],
              ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=0.8)
    ax.text(1.0, -0.2, 'One-way ANOVA from summary statistics', transform=ax.transAxes, ha='right', fontsize=5, color=MUTED)
    label(ax, 'b', x=-0.04, y=1.06)
    # c CNS precision
    ax = fig.add_subplot(gs[2])
    cp = pd.read_csv('/home/claude/data/cns_precision.csv')
    for i, d in enumerate(CNS):
        for k, t in enumerate(['Repeatability', 'Day-to-day']):
            r = cp[(cp.drug == d) & (cp.type == t)].iloc[0]
            ax.barh(i + (k - 0.5) * 0.36, r.rsd, height=0.32, color=ORANGE if k == 0 else 'white', ec=ORANGE, lw=0.6)
            ax.text(r.rsd + 0.04, i + (k - 0.5) * 0.36, f'{r.rsd:.2f}', va='center', fontsize=5, color=INK2)
    ax.axvline(2, color=INK2, lw=0.5, ls=(0, (2, 1.5)))
    ax.set_yticks(range(3), CNS); ax.invert_yaxis(); ax.set_xlim(0, 2.1); ax.set_xlabel('%RSD (n = 3)')
    ax.tick_params(axis='y', length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=ORANGE, ec=ORANGE, lw=0.6, label='Repeatability'), Patch(fc='white', ec=ORANGE, lw=0.6, label='Day-to-day')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=1, handlelength=1)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position('right'); ax.spines['left'].set_visible(True)
    label(ax, 'c', x=-0.04, y=1.06)
    save(fig, 'Fig5_precision_robustness')

# ============ Figure 6: linearity & sensitivity ============
def fig6():
    lin = pd.read_csv('/home/claude/data/linearity.csv').set_index('drug').loc[ALL]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 66 * MM), gridspec_kw=dict(wspace=0.12, left=0.14, right=0.99, top=0.86, bottom=0.17, width_ratios=[1, 1, 1.25]))
    y = np.arange(len(ALL)); cols = [panel_color(d) for d in ALL]
    ax = axs[0]
    ax.axvspan(0.95, 0.99, color='#f6e3dc', lw=0)
    for yi, d, c in zip(y, ALL, cols):
        v = lin.loc[d, 'r2']; ax.hlines(yi, 0.95, v, color=GRID, lw=0.6)
        ax.scatter(v, yi, s=16, color=c if v >= 0.99 else 'white', ec=c, lw=0.9, zorder=3)
    ax.axvline(0.99, color=INK2, lw=0.5, ls=(0, (2, 1.5))); ax.text(0.9902, -0.9, 'R² ≥ 0.990', fontsize=5, color=INK2, va='bottom')
    ax.set_xlim(0.95, 1.002); ax.set_xticks([0.96, 0.98, 1.0]); ax.set_xlabel('Coefficient of determination, R²')
    ax.set_yticks(y, ALL); ax.set_ylim(len(ALL) - 0.4, -1.2); ax.tick_params(axis='y', length=0)
    label(ax, 'a', x=-0.55, y=1.04)
    ax = axs[1]
    ax.axvspan(-5, 5, color='#eef4fc', lw=0)
    for yi, d, c in zip(y, ALL, cols):
        v = lin.loc[d, 'intercept_pct']; vv = np.clip(v, -60, 60)
        ax.hlines(yi, 0, vv, color=GRID, lw=0.6)
        ax.scatter(vv, yi, s=16, color=c if abs(v) <= 5 else 'white', ec=c, lw=0.9, zorder=3, marker='o' if abs(v) <= 60 else '>')
        if abs(v) > 5: ax.text(vv + (3 if vv > 0 else -3), yi, f'{v:+.0f}%', va='center', ha='left' if vv > 0 else 'right', fontsize=5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlim(-75, 75); ax.set_xlabel('y-intercept, % of mid-range response'); ax.set_ylim(len(ALL) - 0.4, -1.2)
    ax.text(0, -0.9, '±5%', fontsize=5, color=INK2, ha='center', va='bottom')
    ax.tick_params(axis='y', labelleft=False, length=0)
    label(ax, 'b', x=-0.03, y=1.04)
    ax = axs[2]
    for yi, d, c in zip(y, ALL, cols):
        lo, hi = lin.loc[d, 'lo'], lin.loc[d, 'hi']
        ax.hlines(yi, lo, hi, color=c, lw=2.2, alpha=0.35)
        if d in LODQ:
            ax.scatter(LODQ[d][0], yi, marker='|', s=26, color=INK, lw=0.9, zorder=3)
            ax.scatter(LODQ[d][1], yi, marker='D', s=9, color=c, ec=INK, lw=0.3, zorder=3)
        else:
            ax.text(hi * 1.3, yi, 'LOD/LOQ n.r.', va='center', fontsize=4.8, color=MUTED)
    ax.set_xscale('log'); ax.set_xlim(0.01, 200); ax.set_ylim(len(ALL) - 0.4, -1.2)
    ax.set_xlabel('Concentration (µg mL$^{-1}$)'); ax.tick_params(axis='y', labelleft=False, length=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=2.2, alpha=0.5, label='Calibration range'),
                       Line2D([], [], marker='|', ls='', color=INK, ms=5, label='LOD'),
                       Line2D([], [], marker='D', ls='', mfc=MUTED, mec=INK, mew=0.3, ms=3, label='LOQ')],
              ncol=3, loc='lower left', bbox_to_anchor=(0, 0.995), handlelength=1.2, columnspacing=0.8)
    label(ax, 'c', x=-0.03, y=1.04)
    for a in axs: a.axhline(8.5, color=MUTED, lw=0.4, ls=':')
    save(fig, 'Fig6_linearity_sensitivity')

# ============ Audit figure (not for submission) ============
def figA():
    ret = pd.read_csv('/home/claude/data/retention.csv').set_index('drug').loc[ALL]
    rsp = pd.read_csv('/home/claude/data/rs_pred.csv')
    lin = pd.read_csv('/home/claude/data/linearity.csv').set_index('drug')
    imp = pd.read_csv('/home/claude/data/ann_importance.csv')
    fig, axs = plt.subplots(1, 4, figsize=(W2, 58 * MM), gridspec_kw=dict(wspace=0.55, left=0.13, right=0.99, top=0.84, bottom=0.2))
    ax = axs[0]; y = np.arange(len(ALL))
    ax.axvspan(-1, 0, color='#f6e3dc', lw=0)
    for yi, d in zip(y, ALL):
        v = ret.loc[d, 'k_f']; c = panel_color(d)
        ax.hlines(yi, 0, v, color=GRID, lw=0.6); ax.scatter(v, yi, s=14, color='white' if v < 1 else c, ec=c, lw=0.8, zorder=3)
    ax.axvline(0, color=INK2, lw=0.5); ax.axvline(1, color=INK2, lw=0.5, ls=(0, (2, 1.5)))
    ax.set_yticks(y, ALL); ax.set_ylim(len(ALL) - 0.5, -0.5); ax.set_xlim(-0.75, 4); ax.tick_params(axis='y', length=0)
    ax.set_xlabel("Retention factor k (stated flow)"); ax.set_title('Elution before t$_0$', pad=4)
    label(ax, 'a', x=-0.9)
    ax = axs[1]
    ax.plot([0, 11], [0, 11], color=MUTED, lw=0.5, ls=(0, (2, 1.5)))
    ax.scatter(rsp.Rs_pred, rsp.Rs_rep, s=14, color=BLUE, ec='white', lw=0.4)
    ax.set_xlim(0, 11); ax.set_ylim(0, 11); ax.set_xlabel('R$_s$ predicted from t$_R$, N'); ax.set_ylabel('R$_s$ reported'); ax.set_title('Resolution check', pad=4)
    ax.set_aspect('equal'); label(ax, 'b', x=-0.3)
    ax = axs[2]
    sub = lin.loc[NINE]
    ax.plot([0.01, 30], [0.01, 30], color=MUTED, lw=0.5, ls=(0, (2, 1.5)))
    ax.errorbar(sub.lod_r, sub.lod_rep, xerr=[sub.lod_r - sub.lod_r_lo, sub.lod_r_hi - sub.lod_r], fmt='o', ms=3.2, color=BLUE, mec='white', mew=0.4, elinewidth=0.6)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.01, 30); ax.set_ylim(0.01, 30); ax.set_aspect('equal')
    ax.set_xlabel('LOD implied by reported r (µg mL$^{-1}$)'); ax.set_ylabel('LOD reported (µg mL$^{-1}$)'); ax.set_title('LOD consistency', pad=4)
    label(ax, 'c', x=-0.3)
    ax = axs[3]
    yy = np.arange(len(imp))
    ax.barh(yy - 0.18, imp.rep_norm, height=0.34, color='white', ec=INK2, lw=0.6, label='Reported')
    ax.barh(yy + 0.18, imp.recalc_norm, height=0.34, color=BLUE, lw=0, label='Raw / max')
    ax.set_yticks(yy, imp['var']); ax.invert_yaxis(); ax.set_xlabel('Normalised importance (%)'); ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(0.45, 1.22), ncol=1); ax.set_title('ANN importance', pad=4); label(ax, 'd', x=-0.45)
    save(fig, 'FigX_audit_diagnostics')

if __name__ == '__main__':
    import os; os.makedirs(OUTD, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, figA): f(); print('ok', f.__name__)
