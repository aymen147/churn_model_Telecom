import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import datetime

mlflow.set_tracking_uri("http://127.0.0.1:5002")


def prepare_data(data_path="Churn_Modelling.csv"):
    df = pd.read_csv(data_path)
    df = df.drop(["Area code", "State"], axis=1)
    for col in ["International plan", "Voice mail plan"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    y = df_imputed["Churn"]
    X = df_imputed.drop(["Churn"], axis=1)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "feature_names.joblib")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(x_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler


def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]  # For ROC AUC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    conf_matrix_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # Generate classification report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Print summary
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return accuracy, conf_matrix_path, "classification_report.txt"


def save_model(model, model_path="model.joblib"):
    joblib.dump(model, model_path)
    print("Model saved successfully.")


def load_model(file_path="model.joblib"):
    return joblib.load(file_path)


def deploy(model, x_test, y_test):

    with mlflow.start_run(run_name="Model_Deployment") as run:
        mlflow.set_tag("stage", "Model Deployment")

        # Log model parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)

        # Save and log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
        mlflow.register_model(model_uri, "random_forest_model")

        # Evaluate the model and log metrics
        accuracy, conf_matrix_path, report_path = evaluate_model(model, x_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(conf_matrix_path)
        mlflow.log_artifact(report_path)

        # Log additional artifacts
        mlflow.log_artifact("requirements.txt")

        print("Model deployed successfully.")


def predict_with_mlflow(model_uri, input_data):
    mlflow.set_tag("stage", "Model Predict")

    # End any active run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    model = mlflow.pyfunc.load_model(model_uri)

    # Ensure input is a 2D array (reshape if needed)
    if isinstance(input_data, np.ndarray):
        input_data = input_data.reshape(1, -1)  # Ensure (1, N) shape
    elif isinstance(input_data, list):
        input_data = np.array(input_data).reshape(1, -1)

    input_df = pd.DataFrame(input_data)  # Convert to DataFrame

    predictions = model.predict(input_df)

    # Start a new run for prediction
    with mlflow.start_run(run_name="Model_Prediction"):
        mlflow.log_param("model_used", model_uri)
        mlflow.log_param("input_shape", input_df.shape)
        mlflow.log_param("Prediction", int(predictions[0]))  # Log a single value
        mlflow.log_param("timestamp", datetime.datetime.now().isoformat())

    return int(predictions[0])  # Return 0 or 1
