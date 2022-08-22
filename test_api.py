import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get_path(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greeting Message": "Thanks for using my App!"}


def test_post_high(client):
    request = client.post("/prediction", json={'age': 40,
                                               'workclass': 'Private',
                                               'education': 'Doctorate',
                                               'marital_status': 'Married-civ-spouse',
                                               'occupation': 'Prof-specialty',
                                               'relationship': 'Not-in-family',
                                               'race': 'White',
                                               'sex': 'Male',
                                               'hours_per_week': 60,
                                               })
    assert request.status_code == 200
    assert request.json() == {"prediction": " >50K"}


def test_post_low(client):
    request = client.post("/prediction", json={'age': 20,
                                               'workclass': 'Private',
                                               'education': 'HS-grad',
                                               'marital_status': 'Never-married',
                                               'occupation': 'Prof-specialty',
                                               'relationship': 'Not-in-family',
                                               'race': 'White',
                                               'sex': 'Male',
                                               'hours_per_week': 30,
                                               })
    assert request.status_code == 200
    assert request.json() == {"prediction": " <=50K"}


def test_post_malformed(client):
    r = client.post("/prediction", json={
        "age": 35,
        "workclass": "None",
        "education": "Some-college",
        "maritalStatus": "None",
        "occupation": "None",
        "relationship": "None",
        "race": "Black",
        "sex": "Male",
        "hours_per_week": 60,
    })
    assert r.status_code == 422
