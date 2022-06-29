"""
Make sample requests based on real data
"""
import numpy as np
import pandas as pd
import requests
from src.utils import download_ya_disk


TEST_DATA_URL = "https://disk.yandex.ru/d/jzNTGEegzNorCg"
TEST_DATA_PATH = "data/test_data.csv"
PREDICT_API_URL = "http://0.0.0.0:5757/predict/"
N_REQUESTS = 10


if __name__ == "__main__":
    download_ya_disk(TEST_DATA_URL, TEST_DATA_PATH, archive=False)
    data = pd.read_csv(TEST_DATA_PATH)
    request_features = list(data.columns)
    for i in np.random.choice(data.shape[0], N_REQUESTS):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            PREDICT_API_URL,
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())
