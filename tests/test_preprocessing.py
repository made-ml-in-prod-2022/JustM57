from typing import Union, List

import numpy as np
import pytest

from src.preprocessing import transform_age, transform_diabetic, transform_general_health, \
    initial_train_features_preprocessing, transform_train_features_linear
from tests.fake_data import fake_features, fake_linear_features


@pytest.mark.parametrize(
    "age_category,expected_age",
    [('80 or older', 80),
     ('18-24', 21)]
)
def test_transform_age(age_category: str, expected_age: float):
    assert transform_age(age_category) == expected_age


@pytest.mark.parametrize(
    "category,value",
    [('Yes', 1),
     ('No', 0),
     ('No, borderline diabetes', [0., 1.]),
     ('Yes (during pregnancy)', [0., 1.])]
)
def test_transform_diabetic(category: str, value: Union[float, List[float]]):
    if isinstance(value, int):
        assert transform_diabetic(category) == value
    else:
        assert (transform_diabetic(category) >= value[0]) and (transform_diabetic(category) <= value[1])


@pytest.mark.parametrize(
    "category",
    [('Poor'), ],
)
def test_transform_general_health(category: str):
    assert isinstance(transform_general_health(category), int)


def test_initial_train_features_preprocessing():
    x_train = fake_features()
    x_train, num_encoder, mean_vals = initial_train_features_preprocessing(x_train)
    assert len(mean_vals) == 9
    assert x_train[mean_vals.index.tolist()].isna().sum().sum() == 0


@pytest.mark.parametrize(
    "scaler_type,one_hot_features",
    [('StandardScaler', ["Race", "Diabetic", "GenHealth"]),
     ('MinMaxScaler', ["Race"])],
)
def test_transform_train_features_linear(scaler_type: str, one_hot_features: List[str]):
    x_train = fake_linear_features()
    x_train, scaler, one_hot_cols, num_cols = transform_train_features_linear(x_train, scaler_type, one_hot_features)
    assert not(str in x_train.dtypes)
    for col in one_hot_cols:
        assert any([c in col for c in one_hot_features])
    if scaler_type == 'MinMaxScaler':
        assert x_train.max().max() <= 1
        assert x_train.min().min() <= 1
    else:
        assert np.abs(x_train.mean().mean()) < 0.1
