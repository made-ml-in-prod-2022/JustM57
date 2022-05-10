"""
inference module
"""
import logging
from typing import NoReturn

import click

from src.preprocessing import process_test_data
from src.train_models import infer_cv_model


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='inference.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)


@click.command()
@click.argument("data_path")
@click.argument("model_path")
@click.argument("predictions_path")
def make_predictions_wrapper(
        data_path: str, model_path: str, predictions_path: str
) -> NoReturn:
    return make_predictions(data_path, model_path, predictions_path)


def make_predictions(
        data_path: str, model_path: str, predictions_path: str
) -> NoReturn:
    """
    main inference pipeline function:
     - load data
     - process_data
     - make predictions
     - save them
    :param data_path:
    :param model_path:
    :param predictions_path:
    :return:
    """
    logger.info('Inference has started')
    x_test = process_test_data(data_path, model_path)
    logger.info('Data has been processed')
    preds = infer_cv_model(x_test, model_path)
    logger.info('Inference is OK')
    preds.to_csv(predictions_path, index=False)
    logger.info('Program has finished')


if __name__ == '__main__':
    make_predictions_wrapper()
