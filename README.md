# 📄 CV Classifier Pro

Système intelligent de classification automatique de CV utilisant Machine Learning.

## 🚀 Quick Start

### Installation
```bash
# Cloner le projet
git clone <votre-repo>
cd cv-classification-mlops

# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle
python scripts/train_model.py

# Lancer l'application
streamlit run app.py
```

### Utilisation de l'API
```bash
# Lancer l'API
cd api
python main.py

# L'API sera disponible sur http://localhost:8000
# Documentation : http://localhost:8000/docs
```

## 📊 Fonctionnalités

- ✅ Classification de CV en 25 catégories
- ✅ Interface web intuitive
- ✅ Upload de fichiers (PDF, DOCX, TXT)
- ✅ API REST
- ✅ Visualisations interactives
- ✅ Historique des prédictions

## 🛠️ Technologies

- Python 3.8+
- Scikit-learn (ML)
- Streamlit (Frontend)
- FastAPI (Backend)
- TF-IDF + KNN

## 📈 Performance

- Accuracy: ~85%
- 25 catégories professionnelles
- Temps de prédiction: <100ms