"""
Processing module
"""
import pickle
import os
import pandas as pd
import click
from typing import Union, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from entities import ProcessingParams, read_training_pipeline_params
from feature_transforms import transform_age, transform_diabetic, transform_general_health


def initial_test_features_preprocessing(
        x_test: pd.DataFrame, num_encoder: OrdinalEncoder,
        mean_vals: pd.Series) -> pd.DataFrame:
    """
    Process categorical features into ordinal and transform on test dataset
    we also encode numerical values and replace missing with mean over column
    :param x_test:
    :param num_encoder:
    :param mean_vals:
    :return:
    """
    x_test['AgeCategory'] = x_test['AgeCategory'].map(transform_age)
    x_test['Diabetic'] = x_test['Diabetic'].map(transform_diabetic)
    x_test['GenHealth'] = x_test['GenHealth'].map(transform_general_health)
    num_features_to_transform = mean_vals.index.tolist()
    x_test.loc[:, num_features_to_transform] = num_encoder.transform(
        x_test[num_features_to_transform])
    x_test.loc[:, num_features_to_transform].fillna(mean_vals, inplace=True)
    return x_test


def transform_test_features_linear(
        x_test: pd.DataFrame, scaler: Union[StandardScaler, MinMaxScaler],
        one_hot_features: List[str], num_features: List[str]) -> pd.DataFrame:
    """
    transform test features based on training strategy:
      -  use predefined dummy columns
      -  use predefined scaler
    :param x_test:
    :param scaler:
    :param one_hot_features:
    :param num_features:
    :return:
    """
    extra_prefix = set()
    for col in one_hot_features:
        prefix, value = col.split('_')
        extra_prefix.add(prefix)
        x_test[col] = x_test[prefix].map(lambda x: int(x == value))
    x_test_one_hot = x_test[one_hot_features]
    x_test_num = x_test[num_features]
    x_test_num = pd.DataFrame(scaler.transform(x_test_num),
                              columns=num_features)
    x_test = pd.concat([x_test_num, x_test_one_hot], axis=1)
    return x_test


def process_test_data(x_test, model_path: str, models_dir: str):
    """
    load pretrained scalers, encoders, params and transform data
    :param test_data_path:
    :param model_path:
    :return:
    """
    with open(model_path, 'rb') as file_in:
        model = pickle.load(file_in)
    # do common transforms
    with open(os.path.join(models_dir, model.processing_params.numerical_encoder_path), 'rb') as file_in:
        (num_encoder, num_mean) = pickle.load(file_in)
    x_test = initial_test_features_preprocessing(x_test, num_encoder, num_mean)
    # specific transforms
    if model.processing_params.processing_type == 'Linear':
        with open(os.path.join(models_dir, model.processing_params.onehot_columns_path), 'rb') as file_in:
            (one_hot_columns, num_columns) = pickle.load(file_in)
        with open(os.path.join(models_dir, model.processing_params.scaler_path), 'rb') as file_in:
            scaler = pickle.load(file_in)
        x_test = transform_test_features_linear(
            x_test, scaler, one_hot_columns, num_columns
        )
    return x_test


@click.command("process_test_data")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--models-dir")
def preprocess(input_dir: str, output_dir, models_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    model_path = os.path.join(models_dir, "logistic_regression.pkl")
    data = process_test_data(data, model_path, models_dir)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)


if __name__ == '__main__':
    preprocess()
