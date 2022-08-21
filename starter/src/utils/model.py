from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)

    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : 
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.'
    """
    preds = model.predict(X)
    return preds

def compute_slice_metrics(data_cat, cat_col, y, preds):
    """ Compute precision, recall, and F1 on model slices and store them in text file.

    Inputs
    ------
    data_cat : pd.DataFrame
        Data with categorical columns.
    X : np.array
        Data used for prediction.
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predictions from the model.
    Returns
    -------
    None.
    """
    with open('starter//src//slice_output.txt', 'w') as f:
        categories = data_cat[cat_col].unique()
        for category in categories:
            category_index = data_cat[data_cat[cat_col]==category].index
            category_precision, category_recall, category_fbeta = compute_model_metrics(y[category_index], preds[category_index])
            
            f.write(f"Category: {category}, Precision: {category_precision}, Recall, {category_recall}, F-Score: {category_fbeta}\n")
