"""
Module de visualisation des données et résultats
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualizer:
    """
    Classe pour créer des visualisations
    """

    def __init__(self, style='seaborn-v0_8-whitegrid', figsize=(12, 6)):
        """
        Initialise le visualizer

        Args:
            style (str): Style matplotlib
            figsize (tuple): Taille par défaut des figures
        """
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("Set2")

    def plot_distributions(self, df, columns=None, save_path=None):
        """
        Affiche les distributions des variables numériques

        Args:
            df (pd.DataFrame): DataFrame
            columns (list): Colonnes à visualiser (None = toutes)
            save_path (str): Chemin pour sauvegarder
        """
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns

        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution de {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Fréquence')
            axes[idx].axvline(df[col].mean(), color='red', linestyle='--',
                            label=f'Moyenne: {df[col].mean():.2f}')
            axes[idx].legend()

        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, df, save_path=None):
        """
        Affiche la matrice de corrélation

        Args:
            df (pd.DataFrame): DataFrame
            save_path (str): Chemin pour sauvegarder
        """
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_boxplots(self, df, columns=None, save_path=None):
        """
        Affiche les boxplots pour détecter les outliers

        Args:
            df (pd.DataFrame): DataFrame
            columns (list): Colonnes à visualiser
            save_path (str): Chemin pour sauvegarder
        """
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns

        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            axes[idx].boxplot(df[col].dropna())
            axes[idx].set_title(f'Boxplot de {col}', fontweight='bold')
            axes[idx].set_ylabel(col)

        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, importance_df, top_n=15, save_path=None):
        """
        Affiche l'importance des variables

        Args:
            importance_df (pd.DataFrame): DataFrame avec colonnes 'Feature' et 'Importance'
            top_n (int): Nombre de features à afficher
            save_path (str): Chemin pour sauvegarder
        """
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} - Importance des Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name='Model', save_path=None):
        """
        Affiche les prédictions vs valeurs réelles

        Args:
            y_true (array): Valeurs réelles
            y_pred (array): Prédictions
            model_name (str): Nom du modèle
            save_path (str): Chemin pour sauvegarder
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2, label='Prédiction parfaite')
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{model_name} - Prédictions vs Valeurs Réelles', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_residuals(self, y_true, y_pred, model_name='Model', save_path=None):
        """
        Affiche la distribution des résidus

        Args:
            y_true (array): Valeurs réelles
            y_pred (array): Prédictions
            model_name (str): Nom du modèle
            save_path (str): Chemin pour sauvegarder
        """
        residuals = y_true - y_pred

        plt.figure(figsize=(12, 5))

        # Histogramme
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Résidus')
        plt.ylabel('Fréquence')
        plt.title(f'{model_name} - Distribution des Résidus', fontweight='bold')

        # Résidus vs prédictions
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prédictions')
        plt.ylabel('Résidus')
        plt.title(f'{model_name} - Résidus vs Prédictions', fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
