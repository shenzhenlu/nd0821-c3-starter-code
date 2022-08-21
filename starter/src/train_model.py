# %%
# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib

import constants
from utils.data import clean_data, process_data
from utils.model import train_model

# Load in data
data = pd.read_csv("starter//data//census.csv")

# %%
# Clean data
data = clean_data(data)

# Train-Test split
train, test = train_test_split(data, test_size=0.3, random_state=87)

# Preprocess data and save preprocessing models
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=constants.cat_features, label="salary", training=True
)

# %%
joblib.dump(encoder, "starter//model//encoder.pkl")
joblib.dump(lb, "starter//model//lb.pkl")

# Train and save ML models
model = train_model(X_train, y_train)
joblib.dump(model, "starter//model//model.pkl")

# Save test data
joblib.dump(test, "starter//data//test.pkl")
