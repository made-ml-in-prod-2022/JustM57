"""
module with helper functions
"""
import os
from urllib.parse import urlencode
from typing import NoReturn

import requests
import pandas as pd
from sklearn.model_selection import train_test_split


try:
    from entities import DataPaths
except ImportError:
    from src.entities import DataPaths


BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'


def download_dataset(data_url: str, data_path: str) -> NoReturn:
    """
    check the existance of the dataset, overwise download it from yandex.disk
    """
    if not os.path.exists(data_path):
        # Получаем загрузочную ссылку
        final_url = BASE_URL + urlencode(dict(public_key=data_url))
        response = requests.get(final_url)
        download_url = response.json()['href']
        # Загружаем файл и сохраняем его
        download_response = requests.get(download_url)
        with open(data_path, 'wb') as file_out:
            file_out.write(download_response.content)


def train_test_dataset(data_paths: DataPaths, test_size=0.1) -> NoReturn:
    """
    Download the data and split it into train and test datasets
    :param data_paths:
    :param test_size:
    :return:
    """
    download_dataset(data_paths.data_url, data_paths.input_data_path)
    dataset = pd.read_csv(data_paths.input_data_path, compression='zip')
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=data_paths.split_random_seed
    )
    train_dataset.to_csv(data_paths.train_data_path, index=False)
    test_dataset.to_csv(data_paths.test_data_path, index=False)
