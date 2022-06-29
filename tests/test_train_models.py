from faker import Faker
from sklearn.metrics import accuracy_score

from src.train_models import train_model_cv
from src.entities import ModelParams, LinearSearchSpace
from tests.fake_data import fake_train


def generate_model_params() -> ModelParams:
    fake = Faker()
    cv_type = fake.random_choices(elements=("StratifiedKFold", "KFold"))[0]
    model_name = 'LogisticRegression'
    search_space = LinearSearchSpace(
        penalty=['l1', 'l2'],
        C=[0.1, 1, 10],
        class_weight=[None, 'Balanced'])
    output_model_path = 'tmp.pkl'
    return ModelParams(cv_type=cv_type, model_name=model_name, search_space=search_space,
                       output_model_path=output_model_path)


def test_train_model_cv():
    x_train, y_train = fake_train()
    params = generate_model_params()
    model = train_model_cv(x_train, y_train, params, categorical_columns=[])
    assert accuracy_score(y_train, model.predict(x_train)) >= 0.51
