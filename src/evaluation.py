"""
Module d'évaluation des modèles
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import pandas as pd


class ModelEvaluator:
    """
    Classe pour évaluer les modèles de régression
    """

    @staticmethod
    def evaluate_regression(y_true, y_pred, model_name="Model"):
        """
        Évalue un modèle de régression

        Args:
            y_true (array): Valeurs réelles
            y_pred (array): Prédictions
            model_name (str): Nom du modèle

        Returns:
            dict: Métriques d'évaluation
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        print(f"\n{'='*60}")
        print(f"ÉVALUATION: {model_name}")
        print(f"{'='*60}")
        print(f"R² Score:     {r2:.4f}")
        print(f"RMSE:         {rmse:.4f}")
        print(f"MAE:          {mae:.4f}")
        print(f"MAPE:         {mape:.2f}%")
        print(f"{'='*60}")

        return {
            'Model': model_name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape
        }

    @staticmethod
    def compare_models(results):
        """
        Compare plusieurs modèles

        Args:
            results (list): Liste de dictionnaires de résultats

        Returns:
            pd.DataFrame: Tableau de comparaison
        """
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("COMPARAISON DES MODÈLES")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

        # Identifier le meilleur modèle
        best_idx = df['R²'].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_r2 = df.loc[best_idx, 'R²']

        print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
        print(f"   R² = {best_r2:.4f}\n")

        return df

    @staticmethod
    def calculate_residuals(y_true, y_pred):
        """
        Calcule les résidus

        Args:
            y_true (array): Valeurs réelles
            y_pred (array): Prédictions

        Returns:
            array: Résidus (y_true - y_pred)
        """
        return y_true - y_pred

    @staticmethod
    def check_overfitting(train_score, test_score, threshold=0.1):
        """
        Vérifie le surapprentissage

        Args:
            train_score (float): Score sur train
            test_score (float): Score sur test
            threshold (float): Seuil de différence acceptable

        Returns:
            dict: Diagnostic
        """
        diff = train_score - test_score
        is_overfitting = diff > threshold

        print(f"\n{'='*60}")
        print("DIAGNOSTIC OVERFITTING")
        print(f"{'='*60}")
        print(f"Score train: {train_score:.4f}")
        print(f"Score test:  {test_score:.4f}")
        print(f"Différence:  {diff:.4f}")

        if is_overfitting:
            print("⚠ ATTENTION: Surapprentissage détecté!")
            print("   Le modèle performe mieux sur train que sur test")
        else:
            print("✓ Pas de surapprentissage significatif")

        print(f"{'='*60}\n")

        return {
            'train_score': train_score,
            'test_score': test_score,
            'difference': diff,
            'is_overfitting': is_overfitting
        }
