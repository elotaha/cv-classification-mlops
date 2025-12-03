"""
Streamlit App – CV Classification

Interface web permettant de tester le classificateur de CV entraîné avec
TF-IDF + KNN. L'application propose :

- classification d'un CV saisi en texte
- upload et analyse de fichiers (PDF, DOCX, TXT)
- classification multiple via CSV
- visualisation des scores de confiance
- historique des prédictions
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Permet d'importer les modules du projet (preprocessing et modèle)
sys.path.append('.')
from src.model.preprocessing import CVPreprocessor
from src.model.classifier import CVClassifier


# ----------------------
# Configuration Streamlit
# ----------------------
st.set_page_config(
    page_title="CV Classifier Pro",
    page_icon="📄",
    layout="wide"
)

# Style visuel rapide
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        text-align: center;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1.2rem;
    }
    .prediction-box {
        padding: 1.8rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------
# Chargement des modèles
# ----------------------
@st.cache_resource
def load_models():
    """Charge les artefacts ML (label encoder, vectorizer, modèle)."""
    try:
        preprocessor = CVPreprocessor()
        preprocessor.load_label_encoder("models/label_encoder.pkl")

        classifier = CVClassifier()
        classifier.load_model(
            "models/cv_classifier.pkl",
            "models/tfidf_vectorizer.pkl"
        )

        return preprocessor, classifier, True

    except Exception as e:
        st.error(f"Impossible de charger les modèles : {e}")
        return None, None, False


# ----------------------
# Fonctions utilitaires
# ----------------------
def predict_with_confidence(text, preproc, clf):
    """Retourne la prédiction principale et les top catégories avec score."""
    cleaned = preproc.clean_resume(text)
    encoded = clf.predict(cleaned)
    label = preproc.label_encoder.inverse_transform([encoded])[0]

    try:
        probas = clf.predict_proba(cleaned)
        top_idx = np.argsort(probas)[-5:][::-1]

        details = [
            {
                "category": preproc.label_encoder.inverse_transform([i])[0],
                "confidence": float(probas[i] * 100)
            }
            for i in top_idx
        ]
    except:
        details = [{"category": label, "confidence": 100.0}]

    return label, details


def confidence_chart(predictions):
    """Graphique horizontal des catégories probables."""
    fig = go.Figure(go.Bar(
        x=[p["confidence"] for p in predictions],
        y=[p["category"] for p in predictions],
        orientation="h",
        text=[f"{p['confidence']:.1f}%" for p in predictions],
        textposition="auto",
        marker=dict(colorscale="Blues")
    ))
    fig.update_layout(
        title="Scores de confiance",
        height=360,
        template="plotly_white"
    )
    return fig


# ----------------------
# Interface principale
# ----------------------
def main():

    st.markdown('<h1 class="main-header">📄 CV Classifier Pro</h1>', unsafe_allow_html=True)
    st.write("Explorez et testez le modèle de classification de CV.")

    preproc, clf, ready = load_models()
    if not ready:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## Informations")
        categories = list(preproc.label_encoder.classes_)
        st.metric("Catégories disponibles", len(categories))

        with st.expander("Afficher la liste complète"):
            for c in sorted(categories):
                st.write("- ", c)

        st.info("""
        Modèle utilisé :
        - Vectorisation TF-IDF
        - KNN multi-classe
        """)

    # Tabs de l'application
    tab_predict, tab_upload, tab_bulk, tab_stats = st.tabs([
        "🔮 Prédiction",
        "📂 Upload fichier",
        "📊 Analyse multiple",
        "📈 Historique"
    ])

    # ----------------------
    # Tab 1 : Prédiction simple
    # ----------------------
    with tab_predict:
        st.subheader("Tester un CV")

        with st.form("predict_form"):
            name = st.text_input("Nom (optionnel)")
            cv_text = st.text_area("Texte du CV", height=260)
            run = st.form_submit_button("Classifier")

        if run:
            if len(cv_text.strip()) < 50:
                st.error("Veuillez entrer au moins 50 caractères.")
            else:
                label, top_preds = predict_with_confidence(cv_text, preproc, clf)

                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{label}</h2>
                    <p>Confiance : {top_preds[0]['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(confidence_chart(top_preds))

                # Historique
                st.session_state.setdefault("history", []).append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name or "Anonyme",
                    "prediction": label,
                    "confidence": top_preds[0]["confidence"]
                })

    # ----------------------
    # Tab 2 : Upload fichier
    # ----------------------
    with tab_upload:
        st.subheader("Analyser un fichier CV")

        file = st.file_uploader("Fichier PDF, DOCX ou TXT", type=["pdf", "docx", "txt"])

        if file:
            try:
                if file.type == "application/pdf":
                    from PyPDF2 import PdfReader
                    text = "".join(page.extract_text() or "" for page in PdfReader(file).pages)

                elif file.type.endswith("wordprocessingml.document"):
                    from docx import Document
                    text = "\n".join(p.text for p in Document(file).paragraphs)

                else:
                    text = file.read().decode("utf-8")

                st.text_area("Aperçu", text[:1000], height=200)

                if st.button("Classifier ce CV"):
                    label, top_preds = predict_with_confidence(text, preproc, clf)
                    st.write(f"### Résultat : **{label}**")

                    st.plotly_chart(confidence_chart(top_preds))

            except Exception as e:
                st.error(f"Erreur lors de la lecture : {e}")

    # ----------------------
    # Tab 3 : Analyse multiple
    # ----------------------
    with tab_bulk:
        st.subheader("Classification en lot (CSV)")

        file = st.file_uploader("CSV avec colonnes 'name' et 'resume'", type="csv")

        if file:
            try:
                df = pd.read_csv(file)

                if not {"name", "resume"}.issubset(df.columns):
                    st.error("Colonnes requises : name, resume")
                else:
                    st.write(df.head())

                    if st.button("Classifier tous les CV"):
                        results = []

                        for _, row in df.iterrows():
                            label, preds = predict_with_confidence(row["resume"], preproc, clf)
                            results.append({
                                "Nom": row["name"],
                                "Catégorie": label,
                                "Confiance": preds[0]["confidence"]
                            })

                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results)

                        st.download_button(
                            "Télécharger résultats",
                            df_results.to_csv(index=False).encode("utf-8"),
                            "resultats.csv",
                            "text/csv"
                        )

            except Exception as e:
                st.error(f"Erreur : {e}")

    # ----------------------
    # Tab 4 : Statistiques
    # ----------------------
    with tab_stats:
        st.subheader("Historique des prédictions")

        hist = st.session_state.get("history", [])

        if hist:
            df_hist = pd.DataFrame(hist)
            st.dataframe(df_hist)

            fig = px.pie(
                df_hist,
                names="prediction",
                title="Répartition des catégories"
            )
            st.plotly_chart(fig)

            if st.button("Effacer l'historique"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.info("Aucune prédiction enregistrée.")

# Lance l'app
if __name__ == "__main__":
    main()
