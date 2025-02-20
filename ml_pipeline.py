from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from model_pipeline import prepare_data, train_model, evaluate_model


def data_preparation():
    """
    Call the data preparation function.
    """
    return prepare_data()


def train():
    """
    Call the training function and return the trained model.
    """
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    return model


def evaluate():
    """
    Call the evaluate function.
    """
    X_train, X_test, y_train, y_test = prepare_data()
    model = train()  # This will return the trained model
    evaluate_model(model, X_test, y_test)


with DAG("ml_pipeline", start_date=datetime(2024, 2, 7), schedule_interval=None) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=data_preparation)
    train = PythonOperator(task_id="train_model", python_callable=train)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate)

    # Set task dependencies
    prepare >> train >> evaluate
