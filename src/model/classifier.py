"""
Module de classification de CV utilisant TF-IDF et un classificateur KNN.

Ce fichier contient une classe permettant d'entraîner un modèle supervisé
pour classer automatiquement des CV en catégories. Le module inclut :
- préparation des données
- vectorisation TF-IDF
- entraînement KNN (One-vs-Rest)
- évaluation complète (métriques, matrice de confusion)
- sauvegarde / chargement du modèle
- prédiction sur nouveaux CV
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class CVClassifier:
    """
    Classificateur de CV basé sur :
    - TF-IDF pour la représentation textuelle,
    - KNN (One-vs-Rest) pour la classification supervisée.

    Ce classificateur fonctionne bien sur des textes courts/moyens
    et permet une baseline solide pour des tâches de classification NLP.
    """

    def __init__(self, max_features=1500, n_neighbors=5):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2)
        )
        self.model = OneVsRestClassifier(
            KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        )
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.is_trained = False

    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prépare les données :
        - séparation train/test
        - vectorisation TF-IDF

        Args:
            df (pd.DataFrame): DataFrame contenant cleaned_resume + catégorie encodée
            test_size (float): ratio du jeu de test
            random_state (int): reproductibilité

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df["cleaned_resume"].values
        y = df["category_encoded"].values

        print("\nPréparation des données…")
        print(f"Nombre d'exemples : {len(X)}")

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print("Vectorisation TF-IDF…")
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Entraîne le classificateur KNN."""
        print("\nEntraînement du modèle…")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        print(f"Accuracy (train) : {train_acc:.4f}")

    def cross_validate(self, X_train, y_train, cv=5):
        """Effectue une validation croisée simple."""
        print(f"\nValidation croisée ({cv}-fold)…")
        scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")
        print("Scores :", scores)
        print("Accuracy moyenne :", scores.mean())
        return scores

    def evaluate(self, X_test, y_test, label_encoder, save_path="../figures"):
        """
        Évalue le modèle :
        - accuracy
        - métriques weighted
        - rapport complet
        - matrice de confusion
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant évaluation.")

        print("\nÉvaluation du modèle…")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy : {accuracy:.4f}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")

        print("\nRapport détaillé :")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        self._plot_confusion_matrix(
            confusion_matrix(y_test, y_pred),
            label_encoder.classes_,
            save_path
        )

        return accuracy, y_pred

    def _plot_confusion_matrix(self, cm, labels, save_path):
        """Affiche et sauvegarde la matrice de confusion."""
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(16, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title("Matrice de confusion")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()

        filepath = os.path.join(save_path, "confusion_matrix.png")
        plt.savefig(filepath, dpi=300)
        print(f"Matrice de confusion sauvegardée : {filepath}")
        plt.close()

    def predict(self, resume_text):
        """Prédit la catégorie encodée d’un CV nettoyé."""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné.")

        X = self.vectorizer.transform([resume_text])
        return self.model.predict(X)[0]

    def predict_proba(self, resume_text):
        """Retourne les probabilités (si disponible)."""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné.")

        X = self.vectorizer.transform([resume_text])

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[0]

        # fallback si KNN ne supporte pas predict_proba
        pred = self.model.predict(X)[0]
        probas = np.zeros(len(self.model.classes_))
        probas[pred] = 1.0
        return probas

    def save_model(self, model_path, vectorizer_path):
        """Sauvegarde le classificateur et le vectorizer."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print("Modèle et vectorizer sauvegardés.")

    def load_model(self, model_path, vectorizer_path):
        """Charge un modèle et son vectorizer."""

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.is_trained = True
        print("Modèle chargé avec succès.")


# Tests manuels
if __name__ == "__main__":
    print("Ce module doit être utilisé avec des données réelles.")
    print("Consultez notebooks/ ou scripts/train_model.py pour un exemple complet.")
