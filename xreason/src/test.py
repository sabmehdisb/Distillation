#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgbooster import XGBooster
from options import Options  # <- ici, on importe Options depuis options.pyimport tempfile
import tempfile
import os

# -----------------------------
# Charger le dataset
# -----------------------------
dataset_path = '../aaai22/bench/divorce/divorce.csv'  # change ton dataset ici
df = pd.read_csv(dataset_path)

# Target = dernière colonne
target_name = df.columns[-1]

X = df.drop(target_name, axis=1)
y = df[target_name]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.95, random_state=42
)

print(f"Entraînement sur {len(X_train)} instances, test sur {len(X_test)} instances")

# -----------------------------
# Entraîner le modèle XGBoost
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
print("Modèle XGBoost entraîné.")

# -----------------------------
# Choisir une instance à expliquer
# -----------------------------
instance = X_test.iloc[0].values
print("Instance à expliquer :", instance)

# -----------------------------
# Créer un répertoire temporaire pour le modèle XReason
# -----------------------------
with tempfile.TemporaryDirectory() as tmpdirname:
    temp_model_path = os.path.join(tmpdirname, 'temp_model.pkl')

    # Sauvegarder le modèle au format XReason (via XGBooster.train)
    # On crée une option minimale pour XGBooster
    soptions = Options([])

    # Créer un fichier temporaire de données au format XReason
    temp_data_path = os.path.join(tmpdirname, 'temp_dataset.csv')
    df.to_csv(temp_data_path, index=False)

    # -----------------------------
    # Charger le modèle via XGBooster
    # -----------------------------
    sxgb = XGBooster(soptions, from_data=temp_data_path)

    # Expliquer l'instance
    soptions.explain = instance.tolist()
    explanation = sxgb.explain(soptions.explain)
    print("Explication :", explanation)
