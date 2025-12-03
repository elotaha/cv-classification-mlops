"""
Tests pour l'application Streamlit
"""
import os
import sys
import pytest
import numpy as np

# Ajouter le dossier racine au sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from src.model.preprocessing import CVPreprocessor
from src.model.classifier import CVClassifier


def test_models_loading():
    """Test du chargement des modèles"""
    preprocessor = CVPreprocessor()
    preprocessor.load_label_encoder('../../models/label_encoder.pkl')
    
    classifier = CVClassifier()
    classifier.load_model(
        '../../models/cv_classifier.pkl',
        '../../models/tfidf_vectorizer.pkl'
    )
    
    assert preprocessor.label_encoder is not None
    assert classifier.is_trained is True

def test_prediction():
    """Test d'une prédiction"""
    preprocessor = CVPreprocessor()
    preprocessor.load_label_encoder('../../models/label_encoder.pkl')
    
    classifier = CVClassifier()
    classifier.load_model(
        '../../models/cv_classifier.pkl',
        '../../models/tfidf_vectorizer.pkl'
    )
    
    test_text = """
    Python developer with 5 years of experience in machine learning.
    Expert in scikit-learn, TensorFlow, and data analysis.
    """
    
    cleaned = preprocessor.clean_resume(test_text)
    prediction = classifier.predict(cleaned)
    
    assert prediction is not None
    assert isinstance(prediction, (int, np.int64))