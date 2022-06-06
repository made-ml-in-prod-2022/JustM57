"""
train module
"""
import logging
from typing import NoReturn

import click

try:
    from utils import train_test_dataset
    from entities import read_training_pipeline_params
    from preprocessing import process_train_data
    from train_models import train_model_cv, save_results
except ImportError:
    from src.utils import train_test_dataset
    from src.entities import read_training_pipeline_params
    from src.preprocessing import process_train_data
    from src.train_models import train_model_cv, save_results


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)


@click.command()
@click.argument("config_path")
def train_pipeline_wrapper(config_path: str):
    return train_pipeline(config_path)


def train_pipeline(config_path: str) -> NoReturn:
    """
    training pipeline function:
      - parse config
      - load dataset
      - process training data
      - train cross validation model
      - save params used for inference
    :param config_path:
    :return:
    """
    logger.info('Training pipeline has started')
    training_params = read_training_pipeline_params(config_path)
    train_test_dataset(training_params.data_paths)
    logger.info('Data is loaded and split for train and test')
    x_train, y_train = process_train_data(
        train_data_path=training_params.data_paths.train_data_path,
        processing_params=training_params.processing_params
    )
    logger.info('Preprocessing has finished')
    cv_model = train_model_cv(
        x_train, y_train, training_params.model_params,
        training_params.processing_params.categorical_features)
    logger.info(f'Model training has finished with score {cv_model.best_score_}')
    logger.info(f'Best params: {cv_model.best_params_}')
    save_results(cv_model, training_params.processing_params,
                 training_params.model_params)
    logger.info("Program has finished")


if __name__ == '__main__':
    train_pipeline_wrapper()
