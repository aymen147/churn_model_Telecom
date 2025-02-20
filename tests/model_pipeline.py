import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn import preprocessing  # type: ignore
import joblib  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from imblearn.combine import SMOTEENN  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def prepare_data(data_path="Churn_Modelling.csv"):
    """
    Prepares the data for modeling.

    Args:
        data_path (str, optional): Path to the CSV data file. Defaults to 'Churn_Modelling.csv'.

    Returns:
        tuple: A tuple containing (X_train_scaled, X_test_scaled, y_train_resampled, y_test).
    """

    # Load the data
    df = pd.read_csv(data_path)

    # Data preprocessing and feature engineering (same as before)
    df = df.drop(["Area code", "State"], axis=1)
    for col in ["International plan", "Voice mail plan"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)
    # ... (rest of the feature engineering code) ...

    # Handle NaN values
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Split data
    y = df_imputed["Churn"]
    X = df_imputed.drop(["Churn"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    # Apply SMOTEENN
    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(x_train, y_train)

    # Scale data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(x_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test


def train_model(x_train, y_train):
    """
    Entraîner un modèle de régression logistique.

    Args:
        x_train (pd.DataFrame): Données d'entraînement.
        y_train (pd.Series): Labels d'entraînement.

    Returns:
        LogisticRegression: Modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Évaluer les performances du modèle sur les données de test.

    Args:
        model (LogisticRegression): Modèle entraîné.
        x_test (pd.DataFrame): Données de test.
        y_test (pd.Series): Labels de test.

    Returns:
        float: Précision du modèle.
    """
    y_prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, model_path="model.joblib"):
    """
    Sauvegarder le modèle entraîné sur disque.

    Args:
        model (LogisticRegression): Modèle entraîné.
        model_path (str): Chemin du fichier où sauvegarder le modèle.
    """
    joblib.dump(model, model_path)
    print("Modèle sauvegardé avec succès.")


def load_model(file_path="model.joblib"):
    """
    Charger un modèle sauvegardé.

    Args:
        file_path (str): Chemin du fichier modèle sauvegardé.

    Returns:
        LogisticRegression: Modèle chargé.
    """
    model = joblib.load(file_path)
    return model
