import logging
import os
from fastapi import FastAPI

from src import utils

logger = logging.getLogger(__name__)


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": utils.hello()}


@app.on_event("startup")
def load_model():
    logger.info('Model has started')
    model_url = os.getenv("MODEL_URL")
    model_path = os.getenv("MODEL_PATH")
    utils.download_model_ya_disk(model_url, model_path)
    logger.info("Successfully loaded model")
