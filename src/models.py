"""
Module de modélisation et entraînement
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib
import time


class ModelTrainer:
    """
    Classe pour entraîner et gérer les modèles de machine learning
    """

    def __init__(self):
        self.models = {}
        self.trained_models = {}

    def add_model(self, name, model):
        """
        Ajoute un modèle à la collection

        Args:
            name (str): Nom du modèle
            model: Instance du modèle sklearn
        """
        self.models[name] = model
        print(f"✓ Modèle '{name}' ajouté")

    def train_model(self, name, X_train, y_train, cv=5):
        """
        Entraîne un modèle spécifique

        Args:
            name (str): Nom du modèle
            X_train: Features d'entraînement
            y_train: Cible d'entraînement
            cv (int): Nombre de folds pour validation croisée

        Returns:
            dict: Résultats de l'entraînement
        """
        if name not in self.models:
            raise ValueError(f"Modèle '{name}' non trouvé")

        model = self.models[name]

        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT: {name}")
        print(f"{'='*60}")

        # Entraînement
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        print(f"✓ Entraînement terminé en {training_time:.2f}s")

        # Validation croisée
        if cv > 0:
            print(f"\nValidation croisée ({cv}-fold)...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            print(f"R² scores CV: {cv_scores}")
            print(f"R² moyen CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Sauvegarder le modèle entraîné
        self.trained_models[name] = model

        return {
            'model': model,
            'training_time': training_time,
            'cv_scores': cv_scores if cv > 0 else None
        }

    def train_all(self, X_train, y_train, cv=5):
        """
        Entraîne tous les modèles

        Args:
            X_train: Features d'entraînement
            y_train: Cible d'entraînement
            cv (int): Nombre de folds pour validation croisée

        Returns:
            dict: Résultats de tous les modèles
        """
        results = {}
        for name in self.models:
            results[name] = self.train_model(name, X_train, y_train, cv)

        return results

    def predict(self, name, X):
        """
        Fait des prédictions avec un modèle entraîné

        Args:
            name (str): Nom du modèle
            X: Features pour prédiction

        Returns:
            array: Prédictions
        """
        if name not in self.trained_models:
            raise ValueError(f"Modèle '{name}' non entraîné")

        return self.trained_models[name].predict(X)

    def save_model(self, name, filepath):
        """
        Sauvegarde un modèle entraîné

        Args:
            name (str): Nom du modèle
            filepath (str): Chemin de sauvegarde
        """
        if name not in self.trained_models:
            raise ValueError(f"Modèle '{name}' non entraîné")

        joblib.dump(self.trained_models[name], filepath)
        print(f"✓ Modèle '{name}' sauvegardé: {filepath}")

    def load_model(self, name, filepath):
        """
        Charge un modèle sauvegardé

        Args:
            name (str): Nom du modèle
            filepath (str): Chemin du fichier
        """
        self.trained_models[name] = joblib.load(filepath)
        print(f"✓ Modèle '{name}' chargé: {filepath}")

    @staticmethod
    def get_feature_importance(model, feature_names):
        """
        Extrait l'importance des variables

        Args:
            model: Modèle entraîné
            feature_names: Noms des features

        Returns:
            pd.DataFrame: Importance des features
        """
        import pandas as pd

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
        elif hasattr(model, 'coef_'):
            return pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
        else:
            print("⚠ Ce modèle ne fournit pas d'importance de features")
            return None
