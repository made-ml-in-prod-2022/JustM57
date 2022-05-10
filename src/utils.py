"""
module with helper functions
"""
import os
from urllib.parse import urlencode
from typing import NoReturn

import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entities import DataPaths


RANDOM_SEED = 57
TRAIN_FILE = 'train_data.csv'
TEST_FILE = 'test_data.csv'
BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
PUBLIC_KEY = 'https://disk.yandex.ru/d/0cGPjd-6nLpFAQ'


def download_dataset(data_path: str) -> NoReturn:
    """
    check the existance of the dataset, overwise download it from yandex.disk
    :param data_path:
    :return:
    """
    if not os.path.exists(data_path):
        # Получаем загрузочную ссылку
        final_url = BASE_URL + urlencode(dict(public_key=PUBLIC_KEY))
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
    download_dataset(data_paths.input_data_path)
    dataset = pd.read_csv(data_paths.input_data_path, compression='zip')
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=RANDOM_SEED
    )
    train_dataset.to_csv(data_paths.train_data_path, index=False)
    test_dataset.to_csv(data_paths.test_data_path, index=False)
