# Deploying a Machine Learning Model on Heroku with FastAPI

Creating a pipeline to train a model and publish it with a public API on Heroku.

## Unit Test

Test suite can be executed by `pytest`

## Instructions

### Model Training
`python starter/src/train_model.py`

### Model Validation with Data Slices

`python starter/src/validate_model.py`

### Serve the API Locally

`uvicorn main:app --reload`


## Requested Files by Rubric


* [Model Card](starter/model_card.md)

* [continuous_development.png](screenshots/continuous_development.png)

* [example.png](screenshots/example.png)

* [live_get.png](screenshots/live_get.png)

* [live_post.png](screenshots/live_post.png)

* [slice_output.txt](starter/src/slice_output.txt)
