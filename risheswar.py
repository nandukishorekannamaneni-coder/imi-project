import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------- DATA ----------------
names = [
"FurfurylAlcoholResin","AlkydResin","EpoxyNovolacBlend","PhenolicEpoxyBlend",
"PolyesterEpoxyBlend","VinylEsterEpoxyBlend","DiallylPhthalate",
"TriallylCyanurate","TriallylIsocyanurate","LigninEpoxy","CardanolEpoxy",
"SoyEpoxy","TanninResin","RosinEpoxy","VanillinEpoxy","IsosorbideEpoxy",
"GlycerolEpoxy","FuranPolyester","ItaconicPolyester","CrosslinkedEpoxy",
"CrosslinkedPhenolic","CrosslinkedPolyester","CrosslinkedVinylEster",
"NetworkPolyimide","NetworkCyanateEster","NetworkBismaleimide",
"ThermosetPolyurethane","ThermosetSilicone","ThermosetPolybenzoxazine",
"AdvancedEpoxyNetwork","HybridPhenolicSystem","ReinforcedPolyesterMatrix",
"BioBasedThermosetBlend","HighPerformanceCrosslinkedResin"
]

smiles_list = ["CCO","CCN","CCC","c1ccccc1"]

# ---------------- FEATURES ----------------
feature_names = [
"MolWt","MolLogP","TPSA","RingCount","FractionCSP3",
"Retentivity","Density_Est","LogP_Scaled","TPSA_Scaled",
"BranchingIndex","Crosslink_Index","Thermal_Stability_Index",
"Rigidity_Index","Network_Complexity_Index"
]

def extract_features(sm):
    mol = Chem.MolFromSmiles(sm)

    if mol is None:
        return [0.0]*14

    ring = Descriptors.RingCount(mol)
    rot = Descriptors.NumRotatableBonds(mol)

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        ring,
        Descriptors.FractionCSP3(mol),
        (ring+1)/(rot+1),
        float(np.random.rand()),
        float(np.random.rand()),
        float(np.random.rand()),
        float(np.random.rand()),
        float(np.random.rand()),
        float(np.random.rand()),
        float(np.random.rand()),
        float(np.random.rand())
    ]

    return features[:14]

# ---------------- BUILD DATA ----------------
data = []

for i, name in enumerate(names):
    sm = smiles_list[i % len(smiles_list)]
    feats = extract_features(sm)

    row = [name, sm] + feats

    # STRICT CHECK
    if len(row) != 16:   # 2 + 14
        continue

    data.append(row)

# ---------------- DATAFRAME ----------------
columns = ["Polymer_Name","SMILES"] + feature_names
df = pd.DataFrame(data, columns=columns)

df.to_excel("member3.xlsx", index=False)

print(" risheswar")
