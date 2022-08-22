# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Used logistic regression classifier for prediction. Default configuration were used for training.
## Intended Use
This model should be used to predict the category of the salary of a person based on it's financials attributes.
## Training Data
Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 70% of the data is used for training.
## Evaluation Data
Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 30% of the data is used to validate the model.
## Metrics
The model was evaluated using Precision: 0.5350454788657036, Recall: 0.8309098462816784, F-Score: 0.6509357200976404
## Ethical Considerations
For Ethical Considerations the metics were also calculated on data slices. This will drive to a model that may potentially discriminate people; 
further investigation before using it should be done.
## Caveats and Recommendations
The data is biased based on gender, data imbalance need to be further investigated.
