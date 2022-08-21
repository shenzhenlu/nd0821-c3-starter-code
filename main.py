#%% Put the code for your API here.
import os
import joblib

from fastapi import FastAPI
import pandas as pd

from config import ModelInput
from src import constants
from src.utils.data import process_data
from src.utils.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

@app.get("/")
async def root():
    return {"Greeting Message": "Thanks for using my App!"}

@app.post("/prediction")
async def predict(model_input: ModelInput):
    # transorm input to dataframe
    df_input = pd.DataFrame(model_input.dict(), index=[0])

    # preprocessing
    encoder = joblib.load("starter//model//encoder.pkl")
    lb = joblib.load("starter//model//lb.pkl")

    X_input, _, _, _ = process_data(df_input,
                                    categorical_features=constants.cat_features,
                                    training=False,
                                    encoder=encoder,
                                    lb=lb
                                    )
    # inferencing
    model = joblib.load("starter//model//model.pkl")

    prediction = inference(model, X_input)

    # map the prediction to corresponding label
    label_prediction = lb.inverse_transform(prediction)[0]

    return {"prediction": label_prediction}

# %%
