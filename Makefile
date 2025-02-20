# Déclaration des variables
PYTHON=python3
ENV_NAME=mlops_env
REQUIREMENTS=requirements.txt

# 1. Configuration de l'environnement
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)

# 2. Qualité du code, formatage automatique du code, sécurité du code, etc.
lint:
	@echo "Vérification du code (formatage, qualité, sécurité)..."
	@. $(ENV_NAME)/bin/activate && black .
	@. $(ENV_NAME)/bin/activate && flake8 .
	@. $(ENV_NAME)/bin/activate && bandit -r .

# 3. Préparation des données
data:
	@echo "Préparation des données..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py --prepare

# 4. Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py --train

# 5. Tests unitaires
test:
	@echo "Exécution des tests..."
	@. $(ENV_NAME)/bin/activate && pytest 

# 6. Déploiement du modèle
deploy:
	@echo "Déploiement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py --deploy

# 7. Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py --evaluate

# 8. Démarrage du serveur Jupyter Notebook
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@. $(ENV_NAME)/bin/activate && jupyter notebook

# 9. Nettoyage des fichiers temporaires
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@rm -rf __pycache__ .pytest_cache .coverage
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete

# 10. Exécution de toutes les étapes CI/CD (pour la validation)
all: setup data train test deploy evaluate run_api

# 12. Démarrage de l'API FastAPI
.PHONY: run_api
run_api:
	@echo "Démarrage de l'API FastAPI..."
	@. $(ENV_NAME)/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run_flask
run_flask:
	@echo "Démarrage de l'interface web Flask..."
	@. $(ENV_NAME)/bin/activate && python app_flask.py

# 13. Exécution de la prédiction avec MLflow
.PHONY: predict
predict:
	@echo "Exécution de la prédiction avec MLflow..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py --predict




# 11. Aide pour les commandes disponibles
help:
	@echo "Commandes disponibles :"
	@echo "  make setup       : Configure l'environnement virtuel et installe les dépendances."
	@echo "  make lint        : Vérifie la qualité du code (formatage, qualité, sécurité)."
	@echo "  make data        : Prépare les données pour l'entraînement."
	@echo "  make train       : Entraîne le modèle."
	@echo "  make test        : Exécute les tests unitaires."
	@echo "  make deploy      : Déploie le modèle."
	@echo "  make evaluate    : Évalue le modèle."
	@echo "  make notebook    : Démarre Jupyter Notebook."
	@echo "  make clean       : Nettoie les fichiers temporaires."
	@echo "  make all         : Exécute toutes les étapes CI/CD (setup, data, train, test, deploy, evaluate)."
