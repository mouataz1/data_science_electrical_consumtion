"""
Module de préparation et transformation des données
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Classe pour le traitement et la préparation des données
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def remove_duplicates(self, df):
        """
        Supprime les doublons du DataFrame

        Args:
            df (pd.DataFrame): DataFrame à nettoyer

        Returns:
            pd.DataFrame: DataFrame sans doublons
        """
        before = len(df)
        df_clean = df.drop_duplicates()
        after = len(df_clean)
        print(f"Doublons supprimés: {before - after} ({(before-after)/before*100:.2f}%)")
        return df_clean

    def handle_missing_values(self, df, strategy='median'):
        """
        Gère les valeurs manquantes

        Args:
            df (pd.DataFrame): DataFrame à traiter
            strategy (str): 'median', 'mean', ou 'mode'

        Returns:
            pd.DataFrame: DataFrame avec valeurs manquantes traitées
        """
        df_clean = df.copy()

        # Variables numériques
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                if strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)

        # Variables catégorielles
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        print(f"✓ Valeurs manquantes traitées (stratégie: {strategy})")
        return df_clean

    def handle_outliers(self, df, columns=None, method='winsorize', factor=1.5):
        """
        Traite les outliers

        Args:
            df (pd.DataFrame): DataFrame à traiter
            columns (list): Colonnes à traiter (None = toutes les colonnes numériques)
            method (str): 'winsorize' ou 'remove'
            factor (float): Facteur IQR (1.5 par défaut)

        Returns:
            pd.DataFrame: DataFrame avec outliers traités
        """
        df_clean = df.copy()

        if columns is None:
            columns = df_clean.select_dtypes(include=['int64', 'float64']).columns

        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            if method == 'winsorize':
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            elif method == 'remove':
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        print(f"✓ Outliers traités (méthode: {method})")
        return df_clean

    def encode_categorical(self, df, binary_cols=None, onehot_cols=None):
        """
        Encode les variables catégorielles

        Args:
            df (pd.DataFrame): DataFrame à encoder
            binary_cols (list): Colonnes binaires pour Label Encoding
            onehot_cols (list): Colonnes pour One-Hot Encoding

        Returns:
            pd.DataFrame: DataFrame avec variables encodées
        """
        df_encoded = df.copy()

        # Label Encoding pour colonnes binaires
        if binary_cols:
            for col in binary_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                    df_encoded = df_encoded.drop(columns=[col])

        # One-Hot Encoding
        if onehot_cols:
            for col in onehot_cols:
                if col in df_encoded.columns:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])

        print(f"✓ Variables catégorielles encodées")
        return df_encoded

    def scale_features(self, X_train, X_test=None):
        """
        Standardise les features

        Args:
            X_train (pd.DataFrame): Données d'entraînement
            X_test (pd.DataFrame): Données de test (optionnel)

        Returns:
            tuple: (X_train_scaled, X_test_scaled) ou X_train_scaled si X_test=None
        """
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=None):
        """
        Sépare les données en train/test

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Variable cible
            test_size (float): Taille de l'ensemble de test
            random_state (int): Seed pour reproductibilité
            stratify: Pour stratification (classification)

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        print(f"✓ Données séparées: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
