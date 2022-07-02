import os
import random
from typing import NoReturn
import click
import numpy as np
import pandas as pd
from faker import Faker


YES_NO_COLS = ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma',
               'KidneyDisease', 'SkinCancer')


def fake_features(n_rows: int) -> pd.DataFrame:
    """
    creates all features as they are presented in real dataset
    """
    fake = Faker()
    fake_data = {
        **{col: fake.random_elements(elements=('Yes', 'No'), length=n_rows, unique=False) for col in YES_NO_COLS},
        'BMI': [fake.pyfloat(min_value=10., max_value=100.) for _ in range(n_rows)],
        'PhysicalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(n_rows)],
        'MentalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(n_rows)],
        'SleepTime': [fake.pyint(min_value=1, max_value=24) for _ in range(n_rows)],
        'Sex':  fake.random_elements(elements=('Male', 'Female'), length=n_rows, unique=False),
        'AgeCategory':  fake.random_elements(elements=('60-64', '65-69', '40-44', '30-34', '18-24', '75-79', '50-54',
                                                       '70-74', '80 or older', '35-39', '45-49', '25-29', '55-59'),
                                             length=n_rows, unique=False),
        'Race':  fake.random_elements(elements=('White', 'Black', 'Hispanic', 'Other', 'Asian',
                                                'American Indian/Alaskan Native'), length=n_rows, unique=False),
        'Diabetic':  fake.random_elements(elements=('No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'),
                                          length=n_rows, unique=False),
        'GenHealth':  fake.random_elements(elements=('Very good', 'Good', 'Excellent', 'Fair', 'Poor'),
                                           length=n_rows, unique=False),
    }
    return pd.DataFrame(fake_data)


def fake_target(n_rows: int) -> pd.Series:
    """
    creates fake target from uniform distribution
    :param n_rows:
    :return:
    """
    fake = Faker()
    fake_y = pd.Series(fake.random_elements(elements=('Yes', 'No'), length=n_rows, unique=False), name='HeartDisease')
    return fake_y


@click.command("update_data")
@click.argument("output_dir")
def update_train_data(output_dir: str) -> NoReturn:
    """
    generates fake features and target
    saves results into files
    """
    size = random.randint(100, 1000)
    data = fake_features(size)
    target = fake_target(size)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, 'target.csv'), index=False)


if __name__ == '__main__':
    update_train_data()
