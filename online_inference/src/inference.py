"""
code source for preprocessing and applying the model
"""
from typing import List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

from .entities import LoadedModel, ModelResponse


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


def process_test_data(x_test: pd.DataFrame, loaded_model: LoadedModel):
    """
    load pretrained scalers, encoders, params and transform data
    """
    # do common transforms
    x_test = initial_test_features_preprocessing(x_test, loaded_model.num_encoder, loaded_model.num_mean)
    # specific transforms
    if loaded_model.model.processing_params.processing_type == 'Linear':
        x_test = transform_test_features_linear(
            x_test, loaded_model.scaler, loaded_model.one_hot_cols, loaded_model.num_cols
        )
    return x_test


def infer_model(x_test: pd.DataFrame, model: RandomizedSearchCV) -> pd.Series:
    """
    Creates a prediction using model("Yes" or "No")
    """
    predictions = pd.Series(model.predict_proba(x_test)[:, 1], dtype=float, name='disease')
    return predictions


def make_predictions(data: List, features: List[str], loaded_model: LoadedModel) -> List[ModelResponse]:
    """
    Main pipeline function:
    - feature processing
    - apply model
    - return predictions
    """
    data = pd.DataFrame(data, columns=features)
    ids = data.ID
    data = data.drop("ID", axis=1)
    data = process_test_data(data, loaded_model)
    preds = infer_model(data, loaded_model.model.best_estimator)
    preds.index = ids
    return [ModelResponse(id=id_, disease_probability=pred) for id_, pred in preds.iteritems()]
