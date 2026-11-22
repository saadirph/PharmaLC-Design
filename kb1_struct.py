"""Attach structures to the SIELC condition knowledge base and test whether structure predicts class / column family."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Fragments
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats
D = '/home/claude/data/'
kb = pd.read_csv(D + 'knowledge_base_clean.csv')
R = pd.read_csv(D + 'kb2_records.csv', usecols=['name', 'smiles.std'])
R['k'] = R.name.str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
mp = R.dropna(subset=['k']).drop_duplicates('k').set_index('k')['smiles.std']
MAN = {'2-Aminobenzoic Acid': 'Nc1ccccc1C(=O)O', '2-Aminobiphenyl': 'Nc1ccccc1-c1ccccc1', '3-Aminobiphenyl': 'Nc1cccc(c1)-c1ccccc1',
       'Atrolactic Acid': 'CC(O)(C(=O)O)c1ccccc1', 'Chloranil': 'O=C1C(Cl)=C(Cl)C(=O)C(Cl)=C1Cl', 'Cumene Hydroperoxide': 'CC(C)(OO)c1ccccc1',
       'Benzenesulphonic Acid': 'OS(=O)(=O)c1ccccc1', 'Cumene': 'CC(C)c1ccccc1', '6-Benzylaminopurine': 'c1ccc(CNc2ncnc3[nH]cnc23)cc1',
       'Isonicotinic Acid': 'OC(=O)c1ccncc1', 'M-Toluidine': 'Cc1cccc(N)c1', '6-Methyl-5-Hepten-2-One': 'CC(C)=CCCC(C)=O',
       '1-Naphthylacetic Acid': 'OC(=O)Cc1cccc2ccccc12', '2-Naphthoxyacetic Acid': 'OC(=O)COc1ccc2ccccc2c1', 'N -Nitrosodiethylamine': 'CCN(CC)N=O',
       'Ptsa': 'Cc1ccc(cc1)S(=O)(=O)O', 'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O', 'Acetylsalicylic Acid': 'CC(=O)Oc1ccccc1C(=O)O', '5-Aminoindole': 'Nc1ccc2[nH]ccc2c1',
       '2-Aminopyridine': 'Nc1ccccn1', '4-Aminopyridine': 'Nc1ccncc1', '4-Aminopyridine/Fampridine': 'Nc1ccncc1', '4-Amino-2-Chloropyridine': 'Nc1ccnc(Cl)c1',
       '4-Amino-3-Chloropyridine': 'Nc1ccncc1Cl', 'Acenaphthene': 'c1cc2CCc3cccc(c1)c23', '4-Acetylbenzoic Acid': 'CC(=O)c1ccc(cc1)C(=O)O',
       'Cyclopropanecarboxylic Acid': 'OC(=O)C1CC1', 'Benzonitrile': 'N#Cc1ccccc1', 'Benzylamine': 'NCc1ccccc1', '4-Carboxybenzaldehyde': 'O=Cc1ccc(cc1)C(=O)O',
       'Butane sulfonic Acid': 'CCCCS(=O)(=O)O', 'Heptane sulfonic Acid': 'CCCCCCCS(=O)(=O)O', 'Lutidine': 'Cc1cccc(C)n1', '3-Methyl Pyrazole': 'Cc1cc[nH]n1',
       '4-Methoxyphenol': 'COc1ccc(O)cc1', 'N-N-Dimethylbenzylamine': 'CN(C)Cc1ccccc1', 'Octanesulfonicacid': 'CCCCCCCCS(=O)(=O)O',
       'P-Tolualdehyde': 'Cc1ccc(C=O)cc1', 'P-Toluic Acid': 'Cc1ccc(cc1)C(=O)O', 'Phenyl Salicylate': 'O=C(Oc1ccccc1)c1ccccc1O', 'Pyridine': 'c1ccncc1',
       '2-Pyridylacetic Acid Hydrochloride': 'OC(=O)Cc1ccccn1', '3-Pyridylacetic Acid Hydrochloride': 'OC(=O)Cc1cccnc1', '4-Pyridylacetic Acid Hydrochloride': 'OC(=O)Cc1ccncc1',
       'Terephthalaldehyde': 'O=Cc1ccc(C=O)cc1', 'Trifluroacetic Acid (Trifluoroacetic Acid)': 'OC(=O)C(F)(F)F', 'Pseudoephedrine': 'CNC(C)C(O)c1ccccc1',
       '17a-Ethynylestradiol': 'C#CC1(O)CCC2C3CCc4cc(O)ccc4C3CCC21C', 'B-Estradiol': 'CC12CCC3c4ccc(O)cc4CCC3C1CCC2O',
       'Bifonazole': 'c1ccc(cc1)C(n1ccnc1)c1ccc(cc1)-c1ccccc1', 'Procainamide Hydrochloride': 'CCN(CC)CCNC(=O)c1ccc(N)cc1',
       'Naproxen Sodium': 'COc1ccc2cc(ccc2c1)C(C)C(=O)O', 'Ractopamine Hydrochloride': 'CC(CCc1ccc(O)cc1)NCC(O)c1ccc(O)cc1',
       'Delta-9-Tetrahydrocannabinol (THC)': 'CCCCCc1cc(O)c2C3C=C(C)CCC3C(C)(C)Oc2c1', 'Tetrahydrocannabinol (THC)': 'CCCCCc1cc(O)c2C3C=C(C)CCC3C(C)(C)Oc2c1'}
kb['k'] = kb.compound.str.lower().str.replace(r'\(.*?\)', '', regex=True).str.replace(r'[^a-z0-9]', '', regex=True)
kb['smiles'] = kb.compound.map(MAN).fillna(kb.k.map(mp))
kb['smiles_source'] = np.where(kb.compound.isin(MAN), 'manual', np.where(kb.smiles.notna(), 'RepoRT/PubChem', ''))
def feats(s):
    m = Chem.MolFromSmiles(s)
    return dict(logP=Crippen.MolLogP(m), TPSA=rdMolDescriptors.CalcTPSA(m), MW=Descriptors.MolWt(m), HBD=rdMolDescriptors.CalcNumHBD(m),
                HBA=rdMolDescriptors.CalcNumHBA(m), aromRings=rdMolDescriptors.CalcNumAromaticRings(m), rotB=rdMolDescriptors.CalcNumRotatableBonds(m),
                acid=Fragments.fr_COO(m) + Fragments.fr_COO2(m) + Fragments.fr_sulfonamd(m) * 0 + len(m.GetSubstructMatches(Chem.MolFromSmarts('S(=O)(=O)[OX2H1]'))),
                base=Fragments.fr_NH2(m) + Fragments.fr_NH1(m) + Fragments.fr_NH0(m) - Fragments.fr_amide(m) + Fragments.fr_pyridine(m) * 0,
                fsp3=rdMolDescriptors.CalcFractionCSP3(m))
F = kb[kb.smiles.notna()].copy()
F = pd.concat([F.reset_index(drop=True), pd.DataFrame([feats(s) for s in F.smiles])], axis=1)
F.to_csv(D + 'kb1_structures.csv', index=False)
out = dict(n_struct=len(F), n_manual=int((F.smiles_source == 'manual').sum()))
y = (F['class'] == 'Hydrophilic').astype(int).values
cols = ['logP', 'TPSA', 'MW', 'HBD', 'HBA', 'aromRings', 'rotB', 'acid', 'base', 'fsp3']
for c in ['logP', 'TPSA', 'MW']:
    a, b = F[F['class'] == 'Hydrophobic'][c], F[F['class'] == 'Hydrophilic'][c]
    out[c] = dict(med_phob=a.median(), med_phil=b.median(), p=stats.mannwhitneyu(a, b).pvalue)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
p_lr = cross_val_predict(lr, F[cols], y, cv=cv, method='predict_proba')[:, 1]
rf = RandomForestClassifier(500, min_samples_leaf=2, random_state=0)
p_rf = cross_val_predict(rf, F[cols], y, cv=cv, method='predict_proba')[:, 1]
out['class_auc_lr'] = roc_auc_score(y, p_lr); out['class_auc_rf'] = roc_auc_score(y, p_rf)
out['class_bacc_rf'] = balanced_accuracy_score(y, p_rf > 0.5)
# column family (Primesep vs other) from structure
yf = (F.family == 'Primesep (mixed-mode)').astype(int).values
p_f = cross_val_predict(rf, F[cols], yf, cv=cv, method='predict_proba')[:, 1]
out['family_auc_rf'] = roc_auc_score(yf, p_f); out['n_primesep'] = int(yf.sum())
# add with class as feature too
F['p_class_rf'] = p_rf; F['p_primesep_rf'] = p_f
F.to_csv(D + 'kb1_structures.csv', index=False)
json.dump(out, open(D + 'kb1_struct_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
