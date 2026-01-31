"""
Script d'exemple montrant comment utiliser les modules du projet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Import des modules personnalisés
from src.data_processing import DataProcessor
from src.visualization import Visualizer
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator


def main():
    """
    Exemple complet d'utilisation des modules
    """

    print("="*80)
    print("EXEMPLE D'UTILISATION DES MODULES")
    print("="*80)
    print("\n1. Chargement des données...")

    df = pd.read_csv('data/raw/electrical_consumption.csv')
    print(f"   Dataset chargé: {df.shape}")

    print("\n2. Préparation des données...")

    processor = DataProcessor()

    df = processor.remove_duplicates(df)

    df = processor.handle_missing_values(df, strategy='median')

    df = processor.handle_outliers(df, method='winsorize')

    print("\n3. Séparation features/cible...")

    target_column = 'consommation'

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = processor.split_data(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    print("\n4. Visualisations...")

    viz = Visualizer()

    print("\n5. Entraînement des modèles...")

    trainer = ModelTrainer()

    # Ajouter les modèles
    trainer.add_model('Linear Regression', LinearRegression())
    trainer.add_model('Random Forest', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    ))
    trainer.add_model('Gradient Boosting', GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))

    results = trainer.train_all(X_train_scaled, y_train, cv=5)

    print("\n6. Évaluation des modèles...")

    evaluator = ModelEvaluator()
    evaluation_results = []

    for model_name in ['Linear Regression', 'Random Forest', 'Gradient Boosting']:
        # Prédictions
        y_pred = trainer.predict(model_name, X_test_scaled)

        # Évaluation
        metrics = evaluator.evaluate_regression(
            y_test,
            y_pred,
            model_name=model_name
        )
        evaluation_results.append(metrics)

    print("\n7. Comparaison des modèles...")

    comparison_df = evaluator.compare_models(evaluation_results)

    # Sauvegarder la comparaison
    comparison_df.to_csv('reports/models_comparison.csv', index=False)
    print("   ✓ Comparaison sauvegardée dans 'reports/models_comparison.csv'")

    print("\n8. Analyse de l'importance des variables...")

    # Pour Random Forest (exemple)
    rf_model = trainer.trained_models['Random Forest']
    importance_df = ModelTrainer.get_feature_importance(
        rf_model,
        X_train_scaled.columns
    )

    if importance_df is not None:
        print("\nTop 5 variables importantes:")
        print(importance_df.head())

    print("\n9. Sauvegarde des modèles...")

    for model_name in trainer.trained_models:
        filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        trainer.save_model(model_name, filename)

    # Sauvegarder le scaler
    import joblib
    joblib.dump(processor.scaler, 'models/scaler.pkl')
    print("   ✓ Scaler sauvegardé")

    print("\n" + "="*80)
    print("RÉSUMÉ FINAL")
    print("="*80)

    best_model_idx = comparison_df['R²'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    best_r2 = comparison_df.loc[best_model_idx, 'R²']

    print(f"\n Meilleur modèle: {best_model}")
    print(f"   R² Score: {best_r2:.4f}")
    print(f"\n Tous les résultats ont été sauvegardés!")
    print(f"   - Modèles: models/")
    print(f"   - Comparaison: reports/models_comparison.csv")
    print(f"   - Visualisations: visualizations/")

    print("\n" + "="*80)
    print("SCRIPT TERMINÉ AVEC SUCCÈS!")
    print("="*80)


if __name__ == "__main__":
    """
    Point d'entrée du script

    Usage:
        python exemple_utilisation.py
    """
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n ERREUR: Fichier non trouvé")
        print(f"   {e}")
        print(f"\n Assurez-vous d'avoir:")
        print(f"   1. Placé votre dataset dans 'data/raw/'")
        print(f"   2. Adapté le nom du fichier dans le script")
        print(f"   3. Vérifié le nom de la colonne cible")
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()
