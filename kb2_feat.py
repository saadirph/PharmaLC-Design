import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi
RDLogger.DisableLog('rdApp.*')
from joblib import Parallel, delayed

CASE = {  # case-study drugs (from the two source manuscripts); SMILES verified by formula + InChIKey below
    'Amlodipine': ('CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC', 'C20H25ClN2O5', 'A'),
    'Bisoprolol': ('CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1', 'C18H31NO4', 'A'),
    'Clarithromycin': ('CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)O)(C)O', 'C38H69NO13', 'A'),
    'Clopidogrel': ('COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1', 'C16H16ClNO2S', 'A'),
    'Folic acid': ('Nc1nc2ncc(CNc3ccc(cc3)C(=O)NC(CCC(=O)O)C(=O)O)nc2c(=O)[nH]1', 'C19H19N7O6', 'A'),
    'Glimepiride': ('CCC1=C(C)CN(C1=O)C(=O)NCCc1ccc(cc1)S(=O)(=O)NC(=O)NC1CCC(C)CC1', 'C24H34N4O5S', 'A'),
    'Isosorbide mononitrate': ('OC1COC2C(COC12)O[N+](=O)[O-]', 'C6H9NO6', 'A'),
    'Paracetamol': ('CC(=O)Nc1ccc(O)cc1', 'C8H9NO2', 'A'),
    'Alprazolam': ('Cc1nnc2CN=C(c3ccccc3)c3cc(Cl)ccc3-n12', 'C17H13ClN4', 'B'),
    'Bromazepam': ('O=C1CN=C(c2ccccn2)c2cc(Br)ccc2N1', 'C14H10BrN3O', 'B'),
    'Methylphenidate': ('COC(=O)C(c1ccccc1)C1CCCCN1', 'C14H19NO2', 'B'),
}
NAMES = [n for n, _ in Descriptors.descList]

def desc(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    out = []
    for n, f in Descriptors.descList:
        try: out.append(f(m))
        except Exception: out.append(np.nan)
    return out

def canon_ik(smi):
    m = Chem.MolFromSmiles(smi); return inchi.MolToInchiKey(m) if m else None

if __name__ == '__main__':
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    case = []
    for k, (s, f, panel) in CASE.items():
        m = Chem.MolFromSmiles(s); fm = CalcMolFormula(m)
        case.append(dict(drug=k, smiles=s, formula=fm, formula_ok=fm == f, inchikey=inchi.MolToInchiKey(m), panel=panel))
    case = pd.DataFrame(case); print(case[['drug', 'formula', 'formula_ok', 'inchikey']])
    case.to_csv('/home/claude/data/case_drugs.csv', index=False)
    R = pd.read_csv('/home/claude/data/kb2_records.csv')
    U = R.drop_duplicates('inchikey.std')[['inchikey.std', 'smiles.std']]
    U = pd.concat([U, case.rename(columns={'inchikey': 'inchikey.std', 'smiles': 'smiles.std'})[['inchikey.std', 'smiles.std']]]).drop_duplicates('inchikey.std')
    D = Parallel(n_jobs=2, batch_size=200)(delayed(desc)(s) for s in U['smiles.std'])
    keep = [i for i, d in enumerate(D) if d is not None]
    X = pd.DataFrame([D[i] for i in keep], columns=NAMES); X.insert(0, 'inchikey', U['inchikey.std'].values[keep])
    X = X.replace([np.inf, -np.inf], np.nan)
    X.to_pickle('/home/claude/data/kb2_desc.pkl'); print(X.shape)
