"""
Processing module
"""
import pickle
import os
from typing import List
import click
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from entities import ProcessingParams, read_training_pipeline_params
from feature_transforms import transform_age, transform_diabetic, transform_general_health


def initial_train_features_preprocessing(x_train: pd.DataFrame) -> (
        pd.DataFrame, OrdinalEncoder, pd.Series):
    """
    process several columns, where it is similar in boosting and linear model
    we also encode numerical values and replace missing with mean over column
    :param x_train:
    :return:
    """
    x_train['AgeCategory'] = x_train['AgeCategory'].map(transform_age)
    x_train['Diabetic'] = x_train['Diabetic'].map(transform_diabetic)
    x_train['GenHealth'] = x_train['GenHealth'].map(transform_general_health)
    num_features_to_transform = [
        col for col in x_train.columns if x_train[col].nunique() == 2
    ]
    num_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=np.nan)
    x_train.loc[:, num_features_to_transform] = num_encoder.fit_transform(
        x_train[num_features_to_transform])
    mean_vals = x_train[num_features_to_transform].mean(axis=0)
    x_train.loc[:, num_features_to_transform].fillna(mean_vals, inplace=True)
    return x_train, num_encoder, mean_vals



def transform_target(value: str) -> int:
    """
    convert string into [0, 1]
    :param value:
    :return:
    """
    return value == 'Yes'


def transform_train_features_linear(x_train: pd.DataFrame, scaler_type: str,
                                    one_hot_features: List[str]):
    """
    transform all features in order to fit linear model after:
      -  categorical columns are dummy encoded
      -  numerical columns are scaled
    :param x_train: initial train dataset
    :param scaler_type: name of scaler to transform numerical
    :param one_hot_features: list of categorical features
    :return: updated dataset and fitted encoder+scaler
    """
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    else:
        raise ValueError('No scaler with such name')
    for col in one_hot_features:
        x_train[col] = x_train[col].astype(str)
    x_train_one_hot = pd.get_dummies(x_train[one_hot_features],
                                     prefix=one_hot_features, drop_first=False)
    x_train_num = x_train.drop(one_hot_features, axis=1)
    x_train_num = pd.DataFrame(scaler.fit_transform(x_train_num),
                               columns=x_train_num.columns)
    x_train = pd.concat([x_train_num, x_train_one_hot], axis=1)
    return x_train, scaler, x_train_one_hot.columns, x_train_num.columns


def process_train_data(
        x_train: pd.DataFrame,
        target: pd.Series,
        processing_params: ProcessingParams,
        models_dir: str
) -> [pd.DataFrame, pd.Series]:
    """
    here we do main transformations and save encoders, scalers,
    other params which are useful during inference
    :param train_data_path:
    :param processing_params:
    :return:
    """
    y_train = target.map(transform_target)
    # do common transforms
    x_train, num_encoder, num_mean = initial_train_features_preprocessing(x_train)
    with open(os.path.join(models_dir, processing_params.numerical_encoder_path), 'wb') as file_out:
        pickle.dump((num_encoder, num_mean), file_out)
    # do specific transforms
    if processing_params.processing_type == 'Linear':
        x_train, scaler, one_hot_columns, num_columns = transform_train_features_linear(
            x_train,
            processing_params.scaler_type,
            processing_params.categorical_features
        )
        with open(os.path.join(models_dir, processing_params.onehot_columns_path), 'wb') as file_out:
            pickle.dump((one_hot_columns, num_columns), file_out)
        with open(os.path.join(models_dir, processing_params.scaler_path), 'wb') as file_out:
            pickle.dump(scaler, file_out)
    return x_train, y_train


@click.command("process_train_data")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--models-dir")
def preprocess(input_dir: str, output_dir, models_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv")).HeartDisease
    config = read_training_pipeline_params("config.yaml")
    os.makedirs(models_dir, exist_ok=True)
    data, target = process_train_data(data, target, config.processing_params, models_dir)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, 'target.csv'), index=False)


if __name__ == '__main__':
    preprocess()
