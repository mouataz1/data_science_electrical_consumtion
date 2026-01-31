
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PRÉPARATION AUTOMATIQUE DES DONNÉES")
print("=" * 80)

print("\n1. Chargement des données...")
df = pd.read_csv('data/raw/electrical_consumption.csv')
print(f"   Dataset chargé: {df.shape}")

print("\n2. Nettoyage des données...")

df = df.drop_duplicates()

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"   Suppression des colonnes catégorielles: {categorical_cols}")
df_numeric = df.drop(columns=categorical_cols)

df_numeric = df_numeric.fillna(df_numeric.median())

print(f"   Données nettoyées: {df_numeric.shape}")

print("\n3. Séparation features/cible...")
target_variable = 'totalkW_mean'

if target_variable not in df_numeric.columns:
    print(f"   ⚠ ERREUR: Variable cible '{target_variable}' non trouvée!")
    print(f"   Colonnes disponibles: {list(df_numeric.columns)}")
    exit(1)

X = df_numeric.drop(columns=[target_variable])
y = df_numeric[target_variable]

print(f"   Features (X): {X.shape}")
print(f"   Cible (y): {y.shape}")

print("\n4. Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   y_test: {y_test.shape}")

print("\n5. Normalisation des données...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print(f"   ✓ Données normalisées")

print("\n6. Sauvegarde des fichiers...")

import os
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False, header=True)
y_test.to_csv('data/processed/y_test.csv', index=False, header=True)

joblib.dump(scaler, 'models/scaler.pkl')

print("   ✓ X_train.csv")
print("   ✓ X_test.csv")
print("   ✓ y_train.csv")
print("   ✓ y_test.csv")
print("   ✓ scaler.pkl")

print("\n" + "=" * 80)
print("PRÉPARATION TERMINÉE AVEC SUCCÈS!")
print("=" * 80)
print(f"""
Résumé:
  - Dataset original: {df.shape[0]} lignes, {df.shape[1]} colonnes
  - Dataset final: {X.shape[0]} lignes, {X.shape[1]} features
  - Variable cible: {target_variable}
  - Train: {len(X_train)} échantillons (80%)
  - Test: {len(X_test)} échantillons (20%)

Fichiers créés dans:
  - data/processed/
  - models/

Vous pouvez maintenant exécuter le notebook:
  → notebooks/04_modelisation.ipynb
""")

print("=" * 80)
