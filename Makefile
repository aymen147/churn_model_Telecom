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

# 11. Démarrage de l'API FastAPI (localement, sans Docker)
.PHONY: run_api
run_api:
	@echo "Démarrage de l'API FastAPI..."
	@. $(ENV_NAME)/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run_flask
run_flask:
	@echo "Démarrage de l'interface web Flask..."
	@. $(ENV_NAME)/bin/activate && python app_flask.py

# 12. Exécution de la prédiction avec MLflow
.PHONY: predict
predict:
	@echo "Exécution de la prédiction avec MLflow..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py --predict

# 13. Aide pour les commandes disponibles
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
	@echo "  make run_api     : Démarre l'API FastAPI localement."
	@echo "  make run_flask   : Démarre l'interface web Flask localement."
	@echo "  make predict     : Exécute une prédiction avec MLflow."

# Docker-specific tasks for Atelier 6, Step VI

# Variables for Docker commands
DOCKER_IMAGE := aymenbenchaaben/aymen_benchaaben_4ds6_mlops  # Adjust to match your image name (or use mlproject if preferred)
DOCKER_TAG := latest
DOCKER_USERNAME := aymenbenchaaben

# 14. Build the Docker image
.PHONY: build_docker
build_docker:
	@echo "Construction de l'image Docker..."
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

# 15. Run the Docker container
.PHONY: run_docker
run_docker:
	@echo "Lancement du conteneur Docker..."
	@docker run -d -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

# 16. Push the Docker image to Docker Hub
.PHONY: push_docker
push_docker:
	@echo "Publication de l'image Docker sur Docker Hub..."
	@docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

# 17. Updated help with Docker commands
.PHONY: help
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
	@echo "  make run_api     : Démarre l'API FastAPI localement."
	@echo "  make run_flask   : Démarre l'interface web Flask localement."
	@echo "  make predict     : Exécute une prédiction avec MLflow."
	@echo "  make build_docker: Construit l'image Docker localement."
	@echo "  make run_docker  : Lance le conteneur Docker sur le port 8000."
	@echo "  make push_docker : Publie l'image Docker sur Docker Hub."