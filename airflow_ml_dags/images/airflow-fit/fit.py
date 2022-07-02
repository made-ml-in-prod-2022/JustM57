"""
ml model module
"""
import os
import pickle
from warnings import simplefilter
from typing import List, NoReturn

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.exceptions import ConvergenceWarning
from entities import ModelParams, ProcessingParams, ModelAttributes, read_training_pipeline_params



ClassificationModel = LogisticRegression
simplefilter("ignore", category=ConvergenceWarning)


def init_base_model(
        model_name: str, random_state: int, learning_rate: float, categorical_columns: List[int]
) -> ClassificationModel:
    """
    Initiation of a model with params
    :param model_name:
    :param categorical_columns:
    :return: ml model to be fitted
    """
    if model_name == 'LogisticRegression':
        return LogisticRegression(random_state=random_state, solver='saga')
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
    model = init_base_model(model_params.model_name, model_params.random_state,
                            model_params.boosting_learning_rate, categorical_columns)
    scorer = make_scorer(f1_score)
    random_search = RandomizedSearchCV(
        model, param_distributions=model_params.search_space.__dict__,
        n_iter=model_params.hp_search_iter, cv=cross_val, scoring=scorer, n_jobs=-1
    )
    cv_model = random_search.fit(x_train, y_train)
    return cv_model


def save_results(
        models_dir: str,
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
    with open(os.path.join(models_dir, model_params.output_model_path), 'wb') as file_out:
        pickle.dump(model_attributes, file_out)


@click.command("fit")
@click.option("--input-dir")
@click.option("--models-dir")
def fit_model(input_dir, models_dir):
    x_train = pd.read_csv(os.path.join(input_dir, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).HeartDisease
    config = read_training_pipeline_params("config.yaml")
    cv_model = train_model_cv(
        x_train, y_train, config.model_params, config.processing_params.categorical_features)
    save_results(models_dir, cv_model, config.processing_params, config.model_params)


if __name__ == '__main__':
    fit_model()
