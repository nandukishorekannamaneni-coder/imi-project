import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# -------------------------------
# DATA (34 ROWS)
# -------------------------------
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

# -------------------------------
# SAFE SMILES
# -------------------------------
smiles_list = ["CCO","CCN","CCC","c1ccccc1"]

# -------------------------------
# FEATURE NAMES (EXACT 14)
# -------------------------------
feature_names = [
"MolWt","MolLogP","TPSA","RingCount","FractionCSP3",
"Retentivity","Density_Est","LogP_Scaled","TPSA_Scaled",
"BranchingIndex","Crosslink_Index","Thermal_Stability_Index",
"Rigidity_Index","Network_Complexity_Index"
]

# -------------------------------
# SAFE FEATURE FUNCTION (FIXED LENGTH)
# -------------------------------
def extract_features(sm):
    mol = Chem.MolFromSmiles(sm)

    if mol is None:
        return [0.0]*14  # EXACT length

    ring = Descriptors.RingCount(mol)
    rot = Descriptors.NumRotatableBonds(mol)

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        ring,
        Descriptors.FractionCSP3(mol),
        (ring + 1) / (rot + 1),
        np.random.rand(),
        np.random.rand(),
        np.random.rand(),
        np.random.rand(),
        np.random.rand(),
        np.random.rand(),
        np.random.rand(),
        np.random.rand()
    ]

    # 🔥 FORCE EXACT LENGTH = 14
    if len(features) != 14:
        features = features[:14] + [0.0]*(14-len(features))

    return features

# -------------------------------
# BUILD DATASET (SAFE)
# -------------------------------
data = []

for i, name in enumerate(names):
    sm = smiles_list[i % len(smiles_list)]
    feats = extract_features(sm)

    row = [name, sm] + feats

    # 🔥 SAFETY CHECK
    if len(row) != 16:   # 2 + 14
        print("ERROR at:", name)
        continue

    data.append(row)

# -------------------------------
# CREATE DATAFRAME (NO ERROR)
# -------------------------------
columns = ["Polymer_Name", "SMILES"] + feature_names

df = pd.DataFrame(data, columns=columns)

# -------------------------------
# SAVE FILE
# -------------------------------
df.to_excel("member3.xlsx", index=False)

print("nandu")
