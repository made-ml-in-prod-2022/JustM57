import os
import pickle
from typing import NoReturn
from urllib.parse import urlencode
from pathlib import Path

import requests
import py7zr

from .entities import LoadedModel, ModelAttributes, ProcessingParams, ModelParams, LinearSearchSpace


BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
KEYS = ('logistic_regression', 'numerical_encoder', 'onehot', 'scaler')


class CustomUnpickler(pickle.Unpickler):
    """
    helps to solve unpickling issues
    """
    def find_class(self, module, name):
        if name == 'ModelAttributes':
            return ModelAttributes
        if name == 'ProcessingParams':
            return ProcessingParams
        if name == 'ModelParams':
            return ModelParams
        if name == 'LinearSearchSpace':
            return LinearSearchSpace
        return super().find_class(module, name)


def download_ya_disk(url: str, path: str, archive=True) -> NoReturn:
    """
    download model/test_data from Yandex disk and extract from archive(if neccesary)
    """
    final_url = BASE_URL + urlencode(dict(public_key=url))
    response = requests.get(final_url)
    if response.status_code != 200:
        raise ConnectionError(f"Issues downloading from {final_url}\nIncorrect response {response.json()['error']}")
    download_url = response.json()['href']
    # Загружаем файл и сохраняем его
    download_response = requests.get(download_url)
    folder = Path(path).parent
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as file_out:
        file_out.write(download_response.content)
    if archive:
        with py7zr.SevenZipFile(path, mode='r') as z:
            z.extractall(folder)


def load_models(model_path: str):
    """
    Unpickles all model files
    """
    model_folder = Path(model_path).parent
    model_files = os.listdir(model_folder)
    model_name, enc_name, onehot_name, scaler_name = [list(filter(lambda x: key in x, model_files))[0] for key in KEYS]
    # load model files
    with open(os.path.join(model_folder, model_name), 'rb') as file_in:
        model = CustomUnpickler(file_in).load()
    with open(os.path.join(model_folder, enc_name), 'rb') as file_in:
        (num_encoder, num_mean) = pickle.load(file_in)
    with open(os.path.join(model_folder, onehot_name), 'rb') as file_in:
        (one_hot_columns, num_columns) = pickle.load(file_in)
    with open(os.path.join(model_folder, scaler_name), 'rb') as file_in:
        scaler = pickle.load(file_in)
    results = LoadedModel(model=model, num_encoder=num_encoder, num_mean=num_mean, one_hot_cols=one_hot_columns,
                          num_cols=num_columns, scaler=scaler)
    return results
