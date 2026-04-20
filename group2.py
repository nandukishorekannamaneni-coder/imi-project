import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_excel('thermoset_polymer_dataset.xlsx', sheet_name='Full Dataset')

features = [
    'crosslink_density', 'n_ester', 'n_epoxy', 'n_triazine', 'n_urethane',
    'n_amide', 'ester_bond_density', 'hydrolysis_rate_const',
    'enzymatic_accessibility', 'crystallinity_pct', 'degradation_rate',
    'biodeg_halflife_days', 'water_absorption', 'chemical_resistance',
    'LogP_rdkit', 'TPSA_rdkit', 'MW_rdkit', 'thermal_stability',
    'Tg', 'elastic_modulus'
]

X = df[features].fillna(df[features].mean())
y_class = df['Biodegradability']
y_score = df['bio_score']

# ==============================
# 2. TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# ==============================
# 3. TRAIN MODELS
# ==============================
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train, y_score.loc[X_train.index])

# ==============================
# 4. EVALUATION
# ==============================
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy (REAL): {accuracy:.3f}")

# ==============================
# 5. INVERSE DESIGN
# ==============================
optimal = df[(df['Biodegradability'] == 1) & (df['bio_score'] >= 0.8)]
recommended = optimal[features].mean()

print("\n--- Inverse Design Target Values ---")
print(recommended)

# ==============================
# 6. TANIMOTO SIMILARITY
# ==============================
generator = GetMorganGenerator(radius=2, fpSize=2048)

def check_similarity(new_smiles, reference_smiles_list):
    new_mol = Chem.MolFromSmiles(new_smiles)
    if new_mol is None:
        return 0

    new_fp = generator.GetFingerprint(new_mol)

    similarities = []
    for ref in reference_smiles_list:
        ref_mol = Chem.MolFromSmiles(ref)
        if ref_mol:
            ref_fp = generator.GetFingerprint(ref_mol)
            sim = DataStructs.TanimotoSimilarity(new_fp, ref_fp)
            similarities.append(sim)

    return np.max(similarities) if similarities else 0

# ==============================
# 7. GRAPH 1: FEATURE IMPORTANCE
# ==============================
importances = clf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure()
feat_imp.head(10).plot(kind='bar')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")

# ==============================
# 8. GRAPH 2: PERMUTATION IMPORTANCE
# ==============================
perm = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

plt.figure()
perm_imp.head(10).plot(kind='bar')
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig("permutation_importance.png")

# ==============================
# 9. GRAPH 3: CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.savefig("confusion_matrix.png")

# ==============================
# 10. GRAPH 4: BIO SCORE DISTRIBUTION
# ==============================
plt.figure()
plt.hist(df['bio_score'], bins=20)
plt.title("Bio Score Distribution")
plt.tight_layout()
plt.savefig("bio_score_distribution.png")

# ==============================
# 11. GRAPH 5: TANIMOTO SIMILARITY (EXAMPLE)
# ==============================
sample_smiles = df['SMILES'].dropna().sample(20, random_state=42).tolist()

similarities = []
for smi in sample_smiles:
    sim = check_similarity(smi, sample_smiles)
    similarities.append(sim)

plt.figure()
plt.hist(similarities, bins=10)
plt.title("Tanimoto Similarity Distribution")
plt.tight_layout()
plt.savefig("tanimoto_similarity.png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# ===============================
# STEP 1: LOAD DATASET
# ===============================
df = pd.read_excel("thermoset_polymer_dataset.xlsx")

# ===============================
# STEP 2: PREPROCESSING
# ===============================
# Drop non-numeric / unnecessary columns
df = df.drop(columns=["Polymer_Name", "SMILES", "Polymer_Type"], errors='ignore')

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# ===============================
# STEP 3: FEATURES & TARGET
# ===============================
X = df.drop(columns=["bio_score", "Biodegradability"])
y = df["Biodegradability"]

# ===============================
# STEP 4: TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Total Data:", len(df))
print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))

# ===============================
# STEP 5: TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# STEP 6: PREDICTION
# ===============================
y_pred = model.predict(X_test)

# ===============================
# STEP 7: EVALUATION
# ===============================
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy, 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# STEP 8: CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ===============================
# STEP 9: FEATURE IMPORTANCE
# ===============================
importances = model.feature_importances_
feature_names = X.columns

# Sort features
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ===============================
# DONE
# ===============================
print("\nAll graphs generated successfully!")

# ==============================
# DONE
# ==============================
best_polymer = df[df['Biodegradability'] == 1].sort_values(
    by='bio_score', ascending=False
).iloc[0]

print(best_polymer)

