import pandas as pd
from faker import Faker


FAKE_N_ROWS = 100
YES_NO_COLS = ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma',
               'KidneyDisease', 'SkinCancer')
NUM_COLS = ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma',
            'KidneyDisease', 'SkinCancer', 'Sex')
TRAIN_COLS = ('BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex',
              'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
              'KidneyDisease', 'SkinCancer')


Faker.seed(57)


def fake_features() -> pd.DataFrame:
    fake = Faker()
    fake_data = {
        **{col: fake.random_elements(elements=('Yes', 'No'), length=FAKE_N_ROWS, unique=False) for col in YES_NO_COLS},
        'BMI': [fake.pyfloat(min_value=10., max_value=100.) for _ in range(FAKE_N_ROWS)],
        'PhysicalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(FAKE_N_ROWS)],
        'MentalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(FAKE_N_ROWS)],
        'SleepTime': [fake.pyint(min_value=1, max_value=24) for _ in range(FAKE_N_ROWS)],
        'Sex':  fake.random_elements(elements=('Male', 'Female'), length=FAKE_N_ROWS, unique=False),
        'AgeCategory':  fake.random_elements(elements=('60-64', '65-69', '40-44', '30-34', '18-24', '75-79', '50-54',
                                                       '70-74', '80 or older', '35-39', '45-49', '25-29', '55-59'),
                                             length=FAKE_N_ROWS, unique=False),
        'Race':  fake.random_elements(elements=('White', 'Black', 'Hispanic', 'Other', 'Asian',
                                                'American Indian/Alaskan Native'), length=FAKE_N_ROWS, unique=False),
        'Diabetic':  fake.random_elements(elements=('No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'),
                                          length=FAKE_N_ROWS, unique=False),
        'GenHealth':  fake.random_elements(elements=('Very good', 'Good', 'Excellent', 'Fair', 'Poor'),
                                           length=FAKE_N_ROWS, unique=False),
    }
    return pd.DataFrame(fake_data)


def fake_linear_features() -> pd.DataFrame:
    fake = Faker()
    fake_data = {
        **{col: [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_N_ROWS)] for col in NUM_COLS},
        'BMI': [fake.pyfloat(min_value=10., max_value=100.) for _ in range(FAKE_N_ROWS)],
        'PhysicalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(FAKE_N_ROWS)],
        'MentalHealth': [fake.pyint(min_value=0, max_value=30) for _ in range(FAKE_N_ROWS)],
        'SleepTime': [fake.pyint(min_value=1, max_value=24) for _ in range(FAKE_N_ROWS)],
        'AgeCategory': [fake.pyfloat(min_value=18., max_value=80.) for _ in range(FAKE_N_ROWS)],
        'Race': fake.random_elements(elements=('White', 'Black', 'Hispanic', 'Other', 'Asian',
                                               'American Indian/Alaskan Native'), length=FAKE_N_ROWS, unique=False),
        'Diabetic': [fake.pyfloat(min_value=0., max_value=1.) for _ in range(FAKE_N_ROWS)],
        'GenHealth': [fake.pyint(min_value=1, max_value=5) for _ in range(FAKE_N_ROWS)],
    }
    return pd.DataFrame(fake_data)


def fake_train() -> (pd.DataFrame, pd.Series):
    fake = Faker()
    fake_x_train = pd.DataFrame(
        {col: [fake.pyfloat(min_value=0., max_value=1.) for _ in range(FAKE_N_ROWS)] for col in TRAIN_COLS}
    )
    fake_y_train = pd.Series([fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_N_ROWS)], name='HeartDisease')
    return fake_x_train, fake_y_train
