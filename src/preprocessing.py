"""
Processing module
"""
import pickle
from typing import List, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
try:
    from entities import ProcessingParams
except ImportError:
    from src.entities import ProcessingParams


TARGET_COLUMN = 'HeartDisease'
DIABETIC_MAPPING = {
    'Yes': 1,
    'No': 0,
    'No, borderline diabetes': 0.5,
    'Yes (during pregnancy)': 0.5
}
GENERAL_HEALTH_MAPPING = {
    'Very good': 4,
    'Good': 3,
    'Excellent': 5,
    'Fair': 2,
    'Poor': 1,
}


def transform_age(age: str) -> float:
    """
    convert age interval into it's mean
    :param age:
    :return: mean age
    """
    if age == '80 or older':
        return 80
    start_age = int(age.split('-')[0])
    end_age = int(age.split('-')[1])
    return (start_age + end_age) / 2


def transform_diabetic(diabetic: str) -> float:
    """
    transform diabet feature into ordinal
    :param diabetic: str
    :return: float ordinal value
    """
    return DIABETIC_MAPPING[diabetic]


def transform_general_health(gen_health: str) -> int:
    """
    transform general heath into ordinal values
    :param gen_health:
    :return:
    """
    return GENERAL_HEALTH_MAPPING[gen_health]


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


# def transform_train_features_boosting(x_train, categorical_features):
#     pass
#
#
def process_train_data(
        train_data_path: str,
        processing_params: ProcessingParams
) -> [pd.DataFrame, pd.Series]:
    """
    here we do main transformations and save encoders, scalers,
    other params which are useful during inference
    :param train_data_path:
    :param processing_params:
    :return:
    """
    data = pd.read_csv(train_data_path)
    y_train = data[TARGET_COLUMN].map(transform_target)
    x_train = data.drop(TARGET_COLUMN, axis=1)
    # do common transforms
    x_train, num_encoder, num_mean = initial_train_features_preprocessing(x_train)
    with open(processing_params.numerical_encoder_path, 'wb') as file_out:
        pickle.dump((num_encoder, num_mean), file_out)
    # do specific transforms
    if processing_params.processing_type == 'Linear':
        x_train, scaler, one_hot_columns, num_columns = transform_train_features_linear(
            x_train,
            processing_params.scaler_type,
            processing_params.categorical_features
        )
        with open(processing_params.onehot_columns_path, 'wb') as file_out:
            pickle.dump((one_hot_columns, num_columns), file_out)
        with open(processing_params.scaler_path, 'wb') as file_out:
            pickle.dump(scaler, file_out)
    return x_train, y_train


def process_test_data(test_data_path: str, model_path: str):
    """
    load pretrained scalers, encoders, params and transform data
    :param test_data_path:
    :param model_path:
    :return:
    """
    x_test = pd.read_csv(test_data_path)
    if TARGET_COLUMN in x_test.columns:
        x_test.drop(TARGET_COLUMN, axis=1, inplace=True)
    with open(model_path, 'rb') as file_in:
        model = pickle.load(file_in)
    # do common transforms
    with open(model.processing_params.numerical_encoder_path, 'rb') as file_in:
        (num_encoder, num_mean) = pickle.load(file_in)
    x_test = initial_test_features_preprocessing(x_test, num_encoder, num_mean)
    # specific transforms
    if model.processing_params.processing_type == 'Linear':
        with open(model.processing_params.onehot_columns_path, 'rb') as file_in:
            (one_hot_columns, num_columns) = pickle.load(file_in)
        with open(model.processing_params.scaler_path, 'rb') as file_in:
            scaler = pickle.load(file_in)
        x_test = transform_test_features_linear(
            x_test, scaler, one_hot_columns, num_columns
        )
    return x_test
