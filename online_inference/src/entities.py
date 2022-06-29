"""
Dataclasses module
"""
from typing import List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, conlist

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


N_FEATURES = 18


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
class ModelAttributes:
    """
    class of cv results
    """
    best_estimator: RandomizedSearchCV
    processing_params: ProcessingParams
    model_params: ModelParams
    best_params: dict


@dataclass()
class LoadedModel:
    """
    results of model unpickling
    """
    model: ModelAttributes
    num_encoder: OrdinalEncoder
    num_mean: pd.Series
    one_hot_cols: List[str]
    num_cols: List[str]
    scaler: MinMaxScaler


class ModelResponse(BaseModel):
    """
    predict response format
    """
    id: str
    disease_probability: float


class Features(BaseModel):
    """
    input for predict
    """
    data: List[conlist(Union[int, float, str, None], min_items=N_FEATURES, max_items=N_FEATURES)]
    features: List[str]
