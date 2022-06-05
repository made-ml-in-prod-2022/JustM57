"""
App module
"""
import logging
import os
from typing import List, Union

from fastapi import FastAPI, Response, status
# Выглядит странно, но какие-то проблемы с импортом в Docker. Не получилось исправить
try:
    from .src.utils import download_ya_disk, load_models
    from .src.entities import ModelResponse, LoadedModel, Features
    from .src.inference import make_predictions
except ImportError:
    from src.utils import download_ya_disk, load_models
    from src.entities import ModelResponse, LoadedModel, Features
    from src.inference import make_predictions


logger = logging.getLogger(__name__)


app = FastAPI()


loaded_model: Union[LoadedModel, None] = None


@app.on_event("startup")
def load_model():
    """
    Downloads the model from yandex disk(if not exists) and opens all model pickled files
    """
    logger.info('Model has started')
    model_url = os.getenv("MODEL_URL")
    model_path = os.getenv("MODEL_PATH")
    if not os.path.exists(model_path):
        download_ya_disk(model_url, model_path, archive=True)
    logger.info("Successfully downloaded model")
    # load model and scalers into memory
    global loaded_model
    loaded_model = load_models(model_path)


@app.get("/health", status_code=200)
def check_model(response: Response):
    """
    alive service
    """
    if loaded_model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return "Model is ready"


@app.get("/predict/", response_model=List[ModelResponse])
def predict(request: Features):
    """
    calls model inference for the input request
    """
    return make_predictions(request.data, request.features, loaded_model)
