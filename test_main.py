import pytest
from model_pipeline import prepare_data, train_model, evaluate_model

def test_prepare_data():
    # Test the data preparation function
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None
    assert scaler is not None

    assert len(X_train) > 0
    assert len(y_train) > 0
    assert len(X_test) > 0
    assert len(y_test) > 0

def test_train_model():
    # Test the model training function
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    model = train_model(X_train, y_train)

    assert model is not None

def test_evaluate_model():
    # Test the model evaluation function
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    model = train_model(X_train, y_train)
    
    accuracy, _, _ = evaluate_model(model, X_test, y_test)  # Fix: Unpack properly

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
