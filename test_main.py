import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from model_pipeline import prepare_data, train_model, evaluate_model, predict_with_mlflow

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests"""
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

@pytest.fixture
def trained_model(sample_data):
    """Fixture to provide a trained model"""
    return train_model(sample_data['X_train'], sample_data['y_train'])

def test_prepare_data_shape():
    """Test the shapes of prepared data"""
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # Check basic assertions
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None
    assert scaler is not None

    # Check shapes
    assert X_train.shape[1] == X_test.shape[1]  # Same number of features
    assert len(y_train) == X_train.shape[0]     # Matching samples and labels
    assert len(y_test) == X_test.shape[0]       # Matching samples and labels

def test_prepare_data_content():
    """Test the content of prepared data"""
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # Check for no missing values
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    
    # Check if data is scaled properly
    assert np.abs(X_train.mean(axis=0)).mean() < 1.0  # Roughly centered
    assert 0.1 < X_train.std(axis=0).mean() < 10.0    # Roughly scaled
    
    # Check label distribution
    assert set(y_train) == {0, 1}  # Binary classification
    assert set(y_test) == {0, 1}   # Binary classification

def test_train_model_output(sample_data):
    """Test the model training function output"""
    model = train_model(sample_data['X_train'], sample_data['y_train'])
    
    # Check model type
    assert isinstance(model, RandomForestClassifier)
    
    # Check model attributes
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert hasattr(model, 'feature_importances_')
    
    # Check if model has been fitted
    assert model.n_features_in_ == sample_data['X_train'].shape[1]

def test_train_model_performance(sample_data):
    """Test the model's basic performance metrics"""
    model = train_model(sample_data['X_train'], sample_data['y_train'])
    
    # Check training predictions
    train_pred = model.predict(sample_data['X_train'])
    train_accuracy = np.mean(train_pred == sample_data['y_train'])
    
    assert train_accuracy > 0.7  # Basic performance threshold

def test_evaluate_model_metrics(trained_model, sample_data):
    """Test the model evaluation function outputs"""
    accuracy, conf_matrix_path, report_path = evaluate_model(
        trained_model, 
        sample_data['X_test'], 
        sample_data['y_test']
    )
    
    # Check metric types and ranges
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    
    # Check file outputs
    assert conf_matrix_path.endswith('.png')
    assert report_path.endswith('.txt')
    
    # Check if files were created
    import os
    assert os.path.exists(conf_matrix_path)
    assert os.path.exists(report_path)

def test_evaluate_model_performance(trained_model, sample_data):
    """Test the model's performance on test data"""
    accuracy, _, _ = evaluate_model(
        trained_model, 
        sample_data['X_test'], 
        sample_data['y_test']
    )
    
    # Check if performance meets minimum requirements
    assert accuracy > 0.7  # Minimum accuracy threshold
    
    # Test predictions
    predictions = trained_model.predict(sample_data['X_test'])
    assert len(predictions) == len(sample_data['y_test'])
    assert set(predictions) == {0, 1}  # Binary classification

@pytest.mark.integration
def test_full_pipeline_integration():
    """Integration test for the full pipeline"""
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, conf_matrix_path, report_path = evaluate_model(model, X_test, y_test)
    
    # Check entire pipeline output
    assert accuracy > 0.7
    assert isinstance(model, RandomForestClassifier)
    assert os.path.exists(conf_matrix_path)
    assert os.path.exists(report_path)

@pytest.mark.integration
def test_edge_cases(trained_model, sample_data):
    """Test edge cases and potential error conditions"""
    # Test with empty input
    with pytest.raises(ValueError):
        train_model(np.array([]), np.array([]))
    
    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        wrong_shape = np.random.rand(10, sample_data['X_train'].shape[1] + 1)
        trained_model.predict(wrong_shape)