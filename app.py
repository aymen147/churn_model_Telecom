from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# Charger le modèle
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"  # Define path for the scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)  # Load the scaler
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle ou du scaler : {str(e)}")

# Définir l'application FastAPI
app = FastAPI()


# Définir la structure des données d'entrée pour la prédiction
class InputData(BaseModel):
    features: List[float]  # Liste de nombres flottants


# Définir la structure pour les hyperparamètres de retraining
class RetrainParams(BaseModel):
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 1


# Endpoint pour la prédiction du churn
@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)
        # Apply scaling to the incoming prediction data
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        churn_result = "Churn" if prediction[0] == 1 else "Not Churn"
        return {"prediction": churn_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint pour réentraîner le modèle
@app.post("/retrain")
def retrain(params: RetrainParams):
    global model, scaler

    try:
        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data()

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train
        )  # Fit and transform on training data
        X_test_scaled = scaler.transform(
            X_test
        )  # Transform test data with the same scaler

        # Créer et entraîner un nouveau modèle avec les hyperparamètres fournis
        hyperparams = params.dict()
        if hyperparams["max_depth"] == -1:
            hyperparams["max_depth"] = None

        new_model = RandomForestClassifier(**hyperparams)
        new_model.fit(X_train_scaled, y_train)

        # Évaluer le modèle
        accuracy = evaluate_model(new_model, X_test_scaled, y_test)

        # Sauvegarder le modèle et le scaler
        joblib.dump(scaler, SCALER_PATH)  # Save the scaler
        save_model(new_model, MODEL_PATH)

        # Mettre à jour le modèle et le scaler en mémoire
        model = new_model
        return {
            "success": True,
            "message": "Modèle réentraîné avec succès",
            "accuracy": accuracy,
            "hyperparameters": hyperparams,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant le réentraînement : {str(e)}"
        )
