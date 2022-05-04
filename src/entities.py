"""
Dataclasses module
"""
from typing import List, Optional, Union
from dataclasses import dataclass, field

import yaml
from marshmallow_dataclass import class_schema
from sklearn.model_selection import RandomizedSearchCV


@dataclass()
class DataPaths:
    """
    initial data paths
    """
    input_data_path: str
    train_data_path: str
    test_data_path: str


@dataclass()
class ProcessingParams:
    """
    define processing parameters
    """
    numerical_encoder_path: str
    processing_type: str
    categorical_features: List[str]
    scaler_type: Optional[str]
    onehot_columns_path: Optional[str]
    scaler_path: Optional[str]


@dataclass()
class LinearSearchSpace:
    """
    class of linear model hyperserach space
    """
    penalty: List[str]
    C: List[float]
    class_weight: List[Union[str, None]]


@dataclass()
class ModelParams:
    """
    class to define parameters belong to model
    """
    cv_type: str
    model_name: str
    search_space: Union[LinearSearchSpace]
    output_model_path: str
    hp_search_iter: int = field(default=20)
    cv_seed: int = field(default=20)
    cv_splits: int = field(default=3)


@dataclass()
class TrainingPipelineParams:
    """
    define config of training pipeline
    """
    data_paths: DataPaths
    processing_params: ProcessingParams
    model_params: ModelParams


@dataclass()
class ModelAttributes:
    """
    class of cv results
    """
    best_estimator: RandomizedSearchCV
    processing_params: ProcessingParams
    model_params: ModelParams
    best_params: dict


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    """
    parse config
    :param path:
    :return:
    """
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
