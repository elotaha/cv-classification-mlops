"""
Module de prétraitement pour les CV.

Ce fichier contient une classe dédiée au nettoyage,
à la préparation et à l'encodage des textes de CV pour
des tâches de classification ou d'analyse automatique.
"""

import re
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CVPreprocessor:
    """
    Outils de prétraitement pour les CV :
    - nettoyage du texte
    - normalisation
    - encodage des catégories
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def clean_resume(self, resume_text: str) -> str:
        """
        Nettoie le texte d’un CV pour le rendre exploitable par un modèle.

        Le nettoyage inclut :
        - suppression des URLs, emails, mentions et hashtags
        - suppression de la ponctuation
        - retrait des caractères spéciaux non ASCII
        - normalisation des espaces
        - conversion en minuscules

        Args:
            resume_text (str): texte brut

        Returns:
            str: texte nettoyé et normalisé
        """
        if not isinstance(resume_text, str):
            return ""

        # Supprimer URLs
        resume_text = re.sub(r"http\S+|www\.\S+", " ", resume_text)

        # Supprimer RT/cc
        resume_text = re.sub(r"\b(RT|cc)\b", " ", resume_text, flags=re.IGNORECASE)

        # Supprimer hashtags
        resume_text = re.sub(r"#\S+", " ", resume_text)

        # Supprimer mentions @
        resume_text = re.sub(r"@\S+", " ", resume_text)

        # Supprimer emails
        resume_text = re.sub(r"\S+@\S+", " ", resume_text)

        # Supprimer ponctuation
        resume_text = re.sub(r"[^\w\s]", " ", resume_text)

        # Supprimer caractères non ASCII
        resume_text = re.sub(r"[^\x00-\x7f]", " ", resume_text)

        # Normaliser espaces
        resume_text = re.sub(r"\s+", " ", resume_text)

        return resume_text.lower().strip()

    def process_dataframe(self, df: pd.DataFrame, text_column="Resume", category_column="Category") -> pd.DataFrame:
        """
        Applique le nettoyage du texte et encode les catégories dans un DataFrame.

        Args:
            df (pd.DataFrame): table contenant les CV
            text_column (str): colonne où se trouve le texte brut
            category_column (str): colonne contenant les catégories

        Returns:
            pd.DataFrame: DataFrame enrichi avec :
                - cleaned_resume
                - category_encoded
        """
        df_processed = df.copy()

        print("🔧 Nettoyage des CV...")
        df_processed["cleaned_resume"] = df_processed[text_column].apply(self.clean_resume)

        empty_count = (df_processed["cleaned_resume"].str.len() == 0).sum()
        if empty_count > 0:
            print(f"⚠️  {empty_count} CV vides après nettoyage.")

        print("🏷️ Encodage des catégories...")
        df_processed["category_encoded"] = self.label_encoder.fit_transform(df_processed[category_column])

        print(f"✅ Préprocessing terminé ({len(df_processed)} CV traités)")
        print(f"📁 Nombre de catégories : {len(self.label_encoder.classes_)}")

        return df_processed

    def get_category_mapping(self) -> dict:
        """
        Retourne un mapping {nom_categorie : id_encodé}.

        Returns:
            dict: correspondance label → entier
        """
        if not hasattr(self.label_encoder, "classes_"):
            raise ValueError("Le label encoder n'a pas encore été entraîné.")

        return {label: idx for idx, label in enumerate(self.label_encoder.classes_)}

    def save_label_encoder(self, filepath: str):
        """
        Sauvegarde le label encoder dans un fichier pickle.

        Args:
            filepath (str): chemin du fichier .pkl
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.label_encoder, f)

        print(f"💾 Label encoder sauvegardé → {filepath}")

    def load_label_encoder(self, filepath: str):
        """
        Charge un label encoder existant.

        Args:
            filepath (str): chemin du fichier pickle
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier introuvable : {filepath}")

        with open(filepath, "rb") as f:
            self.label_encoder = pickle.load(f)

        print(f"📂 Label encoder chargé depuis → {filepath}")


# -------------------------------------------------------------------
# Tests manuels (exécutés seulement si ce fichier est lancé directement)
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DU MODULE DE PREPROCESSING")
    print("=" * 60)

    preprocessor = CVPreprocessor()

    # Exemple simple
    example_text = """
    Skills: Python, Machine Learning, @user check this! #AI
    Visit: http://example.com | Email: test@example.com
    Accents: éàè -- Mentions: RT, cc
    """

    print("\nTexte original:")
    print(example_text)

    cleaned = preprocessor.clean_resume(example_text)
    print("\nTexte nettoyé:")
    print(cleaned)

    # Exemple DataFrame
    print("\n" + "=" * 60)
    print("📊 Test DataFrame")

    sample_df = pd.DataFrame({
        "Resume": [
            "Python developer with ML experience",
            "Mechanical engineer @company #engineering",
            "HR professional http://linkedin.com/profile"
        ],
        "Category": ["Data Science", "Mechanical Engineer", "HR"]
    })

    print("\nDataFrame d’origine:")
    print(sample_df)

    processed = preprocessor.process_dataframe(sample_df)
    print("\nDataFrame traité:")
    print(processed[["Category", "cleaned_resume", "category_encoded"]])

    print("\nMapping catégories:")
    print(preprocessor.get_category_mapping())

    # Sauvegarde test
    print("\n" + "=" * 60)
    print("💾 Test sauvegarde / chargement")

    path = "../models/test_label_encoder.pkl"
    preprocessor.save_label_encoder(path)

    new_pre = CVPreprocessor()
    new_pre.load_label_encoder(path)

    print("➡️ Classes chargées :", new_pre.label_encoder.classes_)
    print("\nTests terminés avec succès 🎉")
