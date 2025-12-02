"""
Test de prédiction du modèle de classification de CV
----------------------------------------------------

Ce script permet de tester facilement le modèle entraîné :

1. Chargement du classificateur + TF-IDF + label encoder
2. Prédictions sur plusieurs exemples de CV représentatifs
3. Affichage du Top 3 des catégories probables
4. Un mode interactif pour tester manuellement avec du texte libre

Assurez-vous d’avoir exécuté le script d’entraînement :
    → python scripts/train_model.py
"""

import sys
from pathlib import Path
import numpy as np

# Ajouter le répertoire racine au path (pour l'import des modules internes)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model.preprocessing import CVPreprocessor
from src.model.classifier import CVClassifier


# ---------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------
def load_models():
    """Charge le label encoder, le vectorizer et le classificateur KNN."""
    print("\n📂 Chargement du modèle et du vectorizer...")

    preprocessor = CVPreprocessor()
    classifier = CVClassifier()

    try:
        preprocessor.load_label_encoder(str(ROOT_DIR / "models" / "label_encoder.pkl"))
        classifier.load_model(
            str(ROOT_DIR / "models" / "cv_classifier.pkl"),
            str(ROOT_DIR / "models" / "tfidf_vectorizer.pkl")
        )
        print("✅ Modèles chargés avec succès\n")
        return preprocessor, classifier

    except Exception as e:
        print(f"❌ Impossible de charger les modèles : {e}")
        print("➡️  Lancez d’abord : python scripts/train_model.py")
        sys.exit(1)


# ---------------------------------------------------------------------
# Exemple de CV pour tests automatiques
# ---------------------------------------------------------------------
def get_sample_resumes():
    """Renvoie un ensemble d’exemples de CV pour tester le modèle."""
    return {
        "Data Science": """
        Senior Data Scientist with experience in deep learning and NLP.
        Strong knowledge of Python, TensorFlow, PyTorch, AWS, Spark and ML pipelines.
        """,

        "Java Developer": """
        Java Developer with solid experience in Spring Boot, Hibernate,
        microservices, REST API design and relational databases.
        """,

        "Mechanical Engineer": """
        Mechanical Engineer skilled in CAD modeling (SolidWorks, CATIA),
        thermodynamics, prototyping and manufacturing processes.
        """,

        "HR": """
        HR specialist with expertise in recruitment, onboarding,
        learning & development, ATS tools and employee relations.
        """,

        "DevOps Engineer": """
        DevOps engineer with strong experience in Docker, Kubernetes,
        CI/CD pipelines, Terraform, AWS and monitoring tools.
        """
    }


# ---------------------------------------------------------------------
# Tests automatiques sur exemples
# ---------------------------------------------------------------------
def run_sample_tests(preprocessor, classifier):
    print("="*60)
    print("🔮 TESTS AUTOMATIQUES SUR EXEMPLES DE CV")
    print("="*60)

    examples = get_sample_resumes()

    for expected_category, resume_text in examples.items():
        print("\n" + "="*60)
        print(f"Test attendu : {expected_category}")
        print("="*60)

        print("\n📝 Extrait du CV :")
        print(resume_text.strip()[:200] + "...")

        # Nettoyage + prédiction
        cleaned = preprocessor.clean_resume(resume_text)
        pred_encoded = classifier.predict(cleaned)
        pred_label = preprocessor.label_encoder.inverse_transform([pred_encoded])[0]

        # Affichage de la prédiction
        print(f"\n🎯 Prédiction : {pred_label}")

        # Affichage du top 3 des catégories probables
        try:
            probas = classifier.predict_proba(cleaned)
            top3 = np.argsort(probas)[-3:][::-1]

            print("\n🔝 Top 3 catégories probables :")
            for idx in top3:
                label = preprocessor.label_encoder.inverse_transform([idx])[0]
                score = probas[idx] * 100
                print(f"   {score:5.2f}% — {label}")

        except Exception:
            pass

        # Vérification simple
        correct = pred_label.lower() in expected_category.lower()
        print("✅ Correct !" if correct else f"❌ Incorrect (attendu : {expected_category})")


# ---------------------------------------------------------------------
# Mode interactif
# ---------------------------------------------------------------------
def interactive_mode(preprocessor, classifier):
    print("\n" + "="*60)
    print("💬 MODE INTERACTIF — TESTEZ VOS CV")
    print("="*60)
    print("Tapez 'quit' pour quitter.\n")

    while True:
        text = input("➡️  Entrez un CV : ").strip()

        if text.lower() == "quit":
            print("\n👋 Fin du mode interactif.")
            break

        if len(text) < 20:
            print("⚠️ Texte trop court. Entrez au moins 20 caractères.\n")
            continue

        cleaned = preprocessor.clean_resume(text)
        pred_encoded = classifier.predict(cleaned)
        pred_label = preprocessor.label_encoder.inverse_transform([pred_encoded])[0]

        print(f"🎯 Catégorie prédite : {pred_label}\n")


# ---------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------
def main():
    print("="*60)
    print("🧪 TEST DE PRÉDICTIONS DU MODÈLE")
    print("="*60)

    preprocessor, classifier = load_models()
    run_sample_tests(preprocessor, classifier)
    interactive_mode(preprocessor, classifier)

    print("\n✅ Tous les tests sont terminés.")


if __name__ == "__main__":
    main()
