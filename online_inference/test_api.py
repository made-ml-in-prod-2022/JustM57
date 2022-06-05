"""
Test the predict function in app
"""
import os
import numpy as np
import pytest
from typing import Dict
from fastapi.testclient import TestClient
from faker import Faker
from .heart_desease_app import app


MODEL_URL="https://disk.yandex.ru/d/qYi-7WjL_fHDuw"
MODEL_PATH="model/model.7z"
YES_NO_COLS = ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma',
               'KidneyDisease', 'SkinCancer')
FAKE_N_ROWS = 1


Faker.seed(57)


def fake_features(n_rows: int = FAKE_N_ROWS) -> Dict:
    """
    Creates fake dataset
    """
    fake = Faker()
    fake_data = {
        **{col: fake.random_elements(elements=('Yes', 'No'), length=n_rows, unique=False) for col in YES_NO_COLS},
        'BMI': [fake.pyfloat(min_value=10., max_value=100.) for _ in range(n_rows)],
        'PhysicalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(n_rows)],
        'MentalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(n_rows)],
        'SleepTime': [fake.pyint(min_value=1, max_value=24) for _ in range(n_rows)],
        'Sex':  fake.random_elements(elements=('Male', 'Female'), length=n_rows, unique=False),
        'AgeCategory':  fake.random_elements(elements=('60-64', '65-69', '40-44', '30-34', '18-24', '75-79', '50-54',
                                                       '70-74', '80 or older', '35-39', '45-49', '25-29', '55-59'),
                                             length=n_rows, unique=False),
        'Race':  fake.random_elements(elements=('White', 'Black', 'Hispanic', 'Other', 'Asian',
                                                'American Indian/Alaskan Native'), length=n_rows, unique=False),
        'Diabetic':  fake.random_elements(elements=('No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'),
                                          length=n_rows, unique=False),
        'GenHealth':  fake.random_elements(elements=('Very good', 'Good', 'Excellent', 'Fair', 'Poor'),
                                           length=n_rows, unique=False),
    }
    return fake_data


@pytest.mark.parametrize(
    'n_rows,expectation', [
        (1, 0.27),
        (2, 0.25),
        (1000, 0.21)
    ]
)
def test_predict(n_rows: int, expectation: float):
    """
    creates fake dataset
    adds an ID to it
    then checks app-predict, including the startup initialization
    the model will be downloaded if it doesn't exist on necessary path
    """
    input_json = fake_features(n_rows)
    input_json['ID'] = np.random.choice(10000, size=n_rows, replace=False).tolist()
    features = list(input_json.keys())
    data = [[input_json[key][idx] for key in features] for idx in range(n_rows)]
    input_json = {
        "data": data,
        "features": features
    }
    os.environ["MODEL_URL"] = MODEL_URL
    os.environ["MODEL_PATH"] = MODEL_PATH
    with TestClient(app) as client:
        response = client.get("/predict/", json=input_json)
        assert response.status_code == 200
        assert 'disease_probability' in response.json()[0]
        assert len(response.json()) == n_rows
        p = [row['disease_probability'] for row in response.json()]
        assert round(float(np.mean(p)), 2) == expectation
