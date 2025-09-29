# Churn ML Pipeline

Pipeline ML complet pour la prédiction de churn client avec XGBoost, Optuna, MLflow et API FastAPI.

## 🚀 Fonctionnalités

- **Pipeline ML**: Chargement → Preprocessing → Features → Entraînement avec Optuna → Évaluation
- **Gestion du déséquilibre**: `scale_pos_weight` automatique
- **Optimisation**: Optuna pour hyperparamètres (optimise le recall)
- **Tracking**: MLflow pour métriques et artefacts
- **API**: FastAPI avec validation Pydantic stricte
- **CI/CD**: GitHub Actions avec build Docker automatique

## 📋 Prérequis

- Python 3.11+
- Docker (optionnel)

## 🛠 Installation

```bash
git clone <repo>
cd churn-ml
pip install -r requirements.txt
```

## 🏃‍♂️ Utilisation

### Pipeline ML complet
```bash
python scripts/run_pipeline.py
```

### API FastAPI
```bash
python -m uvicorn src.app.main:app --reload
```
Docs: http://127.0.0.1:8000/docs

### Docker
```bash
docker build -t churn-ml-api .
docker run -p 8000:8000 churn-ml-api
```

## 📊 Données

Placez votre fichier CSV dans `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## 🔧 Configuration

- **Seuil de décision**: `--threshold 0.3` (défaut: 0.5)
- **Essais Optuna**: `--optuna_trials 50` (défaut: 30)
- **MLflow**: `--mlflow_uri mlruns` pour activer le tracking

## 🐳 CI/CD

Le workflow GitHub Actions se déclenche automatiquement sur push vers `main`:
1. Tests et linting
2. Build Docker
3. Push vers Docker Hub
4. Notification de déploiement

Image finale: `<username>/churn-ml-api:latest`

## 🔐 Configuration Docker Hub

Ajoutez ces secrets dans GitHub (Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME`: votre nom d'utilisateur Docker Hub
- `DOCKERHUB_TOKEN`: votre token d'accès Docker Hub

## 📁 Structure

```
churn-ml/
├── src/
│   ├── data/          # Chargement et preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Entraînement, évaluation, tuning
│   ├── serving/       # Inférence
│   ├── utils/         # Utilitaires (MLflow, validation)
│   └── app/           # API FastAPI
├── scripts/           # Scripts d'orchestration
├── artifacts/         # Modèles sauvegardés
├── mlruns/           # MLflow tracking
└── .github/workflows/ # CI/CD
```
