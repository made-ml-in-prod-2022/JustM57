import os
from typing import NoReturn

import numpy as np
import pytest

from src.train_pipeline import train_pipeline
from src.inference import make_predictions
from tests.fake_data import fake_features


FAKE_DATASET_SIZE = 10000
TMP_CFG_NAME = "cfg.yaml"
TMP_CFG_TEXT = '''data_paths:
    data_url: "https://disk.yandex.ru/d/0cGPjd-6nLpFAQ"
    input_data_path: "tmp/data.csv.zip"
    train_data_path: "tmp/train_data.csv"
    test_data_path: "tmp/test_data.csv"
processing_params:
    numerical_encoder_path: "tmp/numerical_encoder_logreg_minmax.pkl"
    processing_type: "Linear"
    scaler_type: "MinMaxScaler"
    categorical_features:
        - "Race"
    onehot_columns_path: "tmp/onehot_logreg_minmax.pkl"
    scaler_path: "tmp/scaler_logreg_minmax.pkl"
model_params:
    cv_type: "StratifiedKFold"
    cv_splits: 3
    model_name: "LogisticRegression"
    search_space:
        penalty: ["l1", "l2"]
        C: [0.01, 0.1, 1., 10., 100.]
        class_weight: ["Balanced", null]
    output_model_path: "tmp/logistic_regression_minmax.pkl"
'''


def create_cfg(tmpdir) -> NoReturn:
    f_out = tmpdir.join(TMP_CFG_NAME)
    f_out.write(TMP_CFG_TEXT)


def create_fake_dataset(p: float) -> NoReturn:
    x_train = fake_features(FAKE_DATASET_SIZE)
    x_train['HeartDisease'] = np.random.choice(['Yes', 'No'], size=FAKE_DATASET_SIZE, p=[p, (1 - p)])
    x_train.to_csv("tmp/data.csv.zip", compression='zip', index=False)


@pytest.mark.parametrize(
    'p', [(0.1),
          (0.5)]
)
def test_train_test(p, tmpdir):
    create_cfg(tmpdir)
    create_fake_dataset(p)
    train_pipeline(os.path.join(tmpdir, TMP_CFG_NAME))
    make_predictions(data_path="tmp/train_data.csv", model_path="tmp/logistic_regression_minmax.pkl",
                     predictions_path="tmp/predictions.csv")
