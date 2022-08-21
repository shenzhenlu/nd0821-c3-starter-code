# %%
import joblib

import constants
from utils.data import process_data
from utils.model import inference, compute_model_metrics, compute_slice_metrics

# Load preprocess and train models
encoder = joblib.load("..//model//encoder.pkl")
lb = joblib.load("..//model//lb.pkl")
model = joblib.load("..//model//model.pkl")

# Load test data
test = joblib.load("..//data//test.pkl")

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(test,
                                           categorical_features=constants.cat_features,
                                           label="salary",
                                           training=False,
                                           encoder=encoder,
                                           lb=lb
                                           )

# %%
# Model metrics overall
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print("Precision", precision, "Recall", recall, "F-Score", fbeta)

# Categorical test data
data_cat_test = test[constants.cat_features].reset_index(drop=True)

# model metrics on data slices
compute_slice_metrics(data_cat_test, "sex", y_test, preds)
# %%
