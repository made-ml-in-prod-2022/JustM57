"""
ml model module
"""
import pickle
from warnings import simplefilter
from typing import Union, List, NoReturn
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.exceptions import ConvergenceWarning

from src.entities import ModelParams, ProcessingParams, ModelAttributes


RANDOM_STATE = 57
BOOSTING_LEARNING_RATE = 0.03


ClassificationModel = Union[LogisticRegression, LGBMClassifier]
simplefilter("ignore", category=ConvergenceWarning)


def init_base_model(
        model_name: str, categorical_columns: List[int]
) -> ClassificationModel:
    """
    Initiation of a model with params
    :param model_name:
    :param categorical_columns:
    :return: ml model to be fitted
    """
    if model_name == 'LogisticRegression':
        return LogisticRegression(random_state=RANDOM_STATE, solver='saga')
    if model_name == 'LGBMClassifier':
        return LGBMClassifier(
            random_state=RANDOM_STATE,
            learning_rate=BOOSTING_LEARNING_RATE,
            categorical_features=categorical_columns
        )
    raise ValueError("Can't recognize model name")


def train_model_cv(
        x_train: pd.DataFrame, y_train: pd.DataFrame,
        model_params: ModelParams, categorical_columns: List[str]
) -> RandomizedSearchCV:
    """
    main training function where we define:
      - cross validation technique
      - base estimator
      - random search hyperparameters selections
    :param x_train:
    :param y_train:
    :param model_params:
    :param categorical_columns:
    :return: random search results
    """
    object_columns = [t for t in x_train.dtypes if isinstance(t, str)]
    if len(object_columns) > 0:
        raise ValueError(f'Train dataset has {object_columns} as objects')
    if model_params.cv_type == 'StratifiedKFold':
        cross_val = StratifiedKFold(n_splits=model_params.cv_splits,
                             random_state=model_params.cv_seed, shuffle=True)
    else:
        cross_val = KFold(n_splits=model_params.cv_splits,
                   random_state=model_params.cv_seed, shuffle=True)
    categorical_columns = [idx for idx, val in enumerate(x_train.columns)
                           if val in categorical_columns]
    model = init_base_model(model_params.model_name, categorical_columns)
    scorer = make_scorer(f1_score)
    random_search = RandomizedSearchCV(
        model, param_distributions=model_params.search_space.__dict__,
        n_iter=model_params.hp_search_iter, cv=cross_val, scoring=scorer, n_jobs=-1
    )
    cv_model = random_search.fit(x_train, y_train)
    return cv_model


def save_results(
        cv_model: RandomizedSearchCV,
        processing_params: ProcessingParams,
        model_params: ModelParams
) -> NoReturn:
    """
    Save best model and parameters of processing and training
    :param cv_model:
    :param processing_params:
    :param model_params:
    :return:
    """
    model_attributes = ModelAttributes(
        best_estimator=cv_model.best_estimator_,
        processing_params=processing_params,
        model_params=model_params,
        best_params=cv_model.best_params_
    )
    # path = Path(model_params.output_model_path)
    # path = path.with_name(
    #     path.stem + f'{round(cv_model.best_score_, 4)}' + path.suffix
    # )
    with open(model_params.output_model_path, 'wb') as file_out:
        pickle.dump(model_attributes, file_out)


def infer_cv_model(x_test: pd.DataFrame, model_path: str) -> pd.Series:
    """
    Make predictions on test dataset using specific model
    :param x_test: processed test dataframe
    :param model_path:
    :return:
    """
    with open(model_path, 'rb') as file_in:
        model = pickle.load(file_in)
    predictions = pd.Series(model.best_estimator.predict(x_test),
                            index=x_test.index, dtype=int, name='predictions')
    predictions = predictions.map(lambda x: 'Yes' if x == 1 else 'No')
    return predictions
