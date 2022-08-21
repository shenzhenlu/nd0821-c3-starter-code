# %%
import joblib
import sys
sys.path.append('..')
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import constants
from src.utils.data import clean_data, process_data
from src.utils.model import inference


def test_clean_data(request):
    data = pd.read_csv("starter//data//census.csv")
    data_clean = clean_data(data)

    assert set(data_clean.columns) == {"age", "workclass", "education", 
                                        "marital_status", "occupation", "relationship",
                                        "race", "sex", "hours_per_week", "salary"}
    
    # store clean data in cache
    request.config.cache.set('cache_json_data', data_clean.to_json())


def test_process_data(request):
    clean_data = pd.read_json(request.config.cache.get('cache_json_data', None))
    # Train-Test split
    train, test = train_test_split(clean_data, test_size=0.3, random_state=87)

    # Preprocess training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=constants.cat_features, label="salary", training=True)

    assert set(np.unique(y_train)) == {0, 1}
    assert X_train.dtype == 'float64'

    # Preprocess test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=constants.cat_features,
        label="salary", training=False, encoder=encoder, lb=lb)
    
    assert set(np.unique(y_test)).issubset({0, 1})
    assert X_test.dtype == 'float64'
    
    request.config.cache.set('cache_json_data', pd.DataFrame(X_test).to_json())


def test_inference(request):
    X_test = pd.read_json(request.config.cache.get('cache_json_data', None)).to_numpy()
    model = joblib.load("starter//model//model.pkl")
    predictions = inference(model, X_test)

    assert set(np.unique(predictions)).issubset({0, 1})