import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------- DATA ----------------
names = [
"SebacicPolyester","SuccinicPolyester","BisphenolA_VinylEster","Novolac_VinylEster",
"EpoxyBased_VinylEster","AcrylatedEpoxy","MethacrylatedEpoxy","Urethane_VinylEster",
"BrominatedVinylEster","ChlorinatedVinylEster","StyreneModifiedVinylEster",
"PhenolicVinylEster","UreaFormaldehyde","MelamineFormaldehyde",
"BenzoguanamineFormaldehyde","AcetoguanamineResin","UreaMelamineResin",
"ThioureaFormaldehyde","AnilineFormaldehyde","DicyandiamideResin",
"AmineFormaldehyde","MelamineUreaResin","PolyimideResin","BismaleimideResin",
"CyanateEsterResin","Polybenzoxazine","Polybenzimidazole",
"PolyetherimideThermoset","PhthalonitrileResin","SiloxaneResin",
"SiliconeResin","Polysilazane","FuranResin"
]

smiles_list = ["CCO","CCN","CCC","c1ccccc1"]

# ---------------- FEATURES ----------------
feature_names = [
"MolWt","MolLogP","TPSA","HeavyAtomCount","RingCount","FractionCSP3",
"AtomCount","AtomDensity","RingDensity","AromaticRatio",
"MolMR","NumAromaticRings","NumAliphaticRings"
]

def extract_features(sm):
    mol = Chem.MolFromSmiles(sm)

    if mol is None:
        return [0.0]*13

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        mol.GetNumAtoms(),
        mol.GetNumAtoms()/10,
        Descriptors.RingCount(mol)/10,
        Descriptors.MolLogP(mol)/5,
        Descriptors.MolMR(mol),
        float(np.random.rand()),
        float(np.random.rand())
    ]

    return features[:13]

# ---------------- BUILD DATA ----------------
data = []

for i, name in enumerate(names):
    sm = smiles_list[i % len(smiles_list)]
    feats = extract_features(sm)

    row = [name, sm] + feats

    # STRICT CHECK
    if len(row) != 15:   # 2 + 13
        continue

    data.append(row)

# ---------------- DATAFRAME ----------------
columns = ["Polymer_Name","SMILES"] + feature_names
df = pd.DataFrame(data, columns=columns)

df.to_excel("member2.xlsx", index=False)

print("naren")
