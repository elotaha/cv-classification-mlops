"""
Script d'entraînement du modèle de classification de CV
-------------------------------------------------------

Ce script exécute l’ensemble du pipeline :

1. Chargement du dataset
2. Nettoyage et preprocessing des CV
3. Vectorisation TF-IDF
4. Entraînement du modèle KNN multi-classe
5. Validation croisée
6. Évaluation sur un test set
7. Sauvegarde des artefacts (modèle, vectorizer, label encoder)

Le script utilise des chemins robustes (pathlib) pour garantir
un fonctionnement correct sur Windows, macOS et Linux.
"""

import warnings
from pathlib import Path
import sys
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Configuration des chemins du projet
# -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "src" / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Ajout du src/ au path
sys.path.insert(0, str(ROOT_DIR))

# Imports internes
from src.model.preprocessing import CVPreprocessor
from src.model.classifier import CVClassifier


# -------------------------------------------------------------
# Fonction : Chargement du dataset
# -------------------------------------------------------------
def load_dataset():
    """Localise et charge UpdatedResumeDataSet.csv."""
    print("\n📂 [1/6] Chargement du dataset...")

    possible_paths = [
        DATA_DIR / "UpdatedResumeDataSet.csv",
        ROOT_DIR / "src" / "datasets" / "UpdatedResumeDataSet.csv",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"✅ Dataset trouvé : {path}")
            df = pd.read_csv(path, encoding="utf-8")
            print(f"📄 {len(df)} CV chargés")
            print(f"🏷️  {df['Category'].nunique()} catégories détectées\n")
            return df

    print("❌ Impossible de trouver UpdatedResumeDataSet.csv.")
    print("Veuillez le placer dans : src/data/")
    sys.exit(1)


# -------------------------------------------------------------
# Fonction principale
# -------------------------------------------------------------
def main():
    print("=" * 60)
    print("🚀 ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION DE CV")
    print("=" * 60)

    # 1. Chargement
    df = load_dataset()

    # 2. Preprocessing
    print("🧹 [2/6] Nettoyage et préparation...")
    preprocessor = CVPreprocessor()
    df_processed = preprocessor.process_dataframe(df)

    # Sauvegarde du label encoder
    label_path = MODELS_DIR / "label_encoder.pkl"
    preprocessor.save_label_encoder(str(label_path))

    # Aperçu
    print("\n🔡  Aperçu des catégories encodées :")
    mapping = preprocessor.get_category_mapping()
    for k, v in list(mapping.items())[:10]:
        print(f"  {v}: {k}")
    print()

    # 3. Split + TF-IDF
    print("🔀 [3/6] Préparation train/test split...")
    classifier = CVClassifier(max_features=1500, n_neighbors=5)
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_processed)

    # 4. Entraînement
    print("\n🤖 [4/6] Entraînement du modèle...")
    classifier.train(X_train, y_train)

    # 5. Validation croisée
    print("\n📊 [5/6] Validation croisée...")
    try:
        cv_scores = classifier.cross_validate(X_train, y_train, cv=5)
    except Exception as e:
        print("⚠️ La validation croisée n'a pas pu être effectuée :", e)
        cv_scores = None

    # 6. Évaluation finale
    print("\n🏁 [6/6] Évaluation sur le test set...")
    accuracy, _ = classifier.evaluate(
        X_test,
        y_test,
        preprocessor.label_encoder,
        save_path=str(FIGURES_DIR)
    )

    # 7. Sauvegarde des artefacts
    print("\n💾 Sauvegarde des artefacts...")
    classifier.save_model(
        str(MODELS_DIR / "cv_classifier.pkl"),
        str(MODELS_DIR / "tfidf_vectorizer.pkl")
    )

    # Résumé final
    print("\n" + "=" * 60)
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 60)

    print(f"📈 Accuracy test set     : {accuracy * 100:.2f}%")
    if cv_scores is not None:
        print(f"📈 Validation croisée    : {cv_scores.mean() * 100:.2f}%")

    print("\n📁 Artefacts générés :")
    print("  - models/cv_classifier.pkl")
    print("  - models/tfidf_vectorizer.pkl")
    print("  - models/label_encoder.pkl")
    print("  - figures/confusion_matrix.png")
    print("=" * 60)
    print("\nFin de l'entraînement.\n")


# -------------------------------------------------------------
# Point d'entrée
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
