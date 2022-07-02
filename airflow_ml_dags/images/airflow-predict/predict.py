import os
import pickle
import click
import pandas as pd
from entities import ModelParams, ProcessingParams, ModelAttributes, read_training_pipeline_params


def infer_cv_model(x_test: pd.DataFrame, model_path: str) -> pd.Series:
    """
    Make predictions on test dataset using specific model
    :param x_test: processed test dataframe
    :param model_path:
    :return:
    """
    with open(model_path, 'rb') as file_in:
        model = pickle.load(file_in)
    predictions = pd.Series(model.best_estimator.predict(x_test),
                            index=x_test.index, dtype=int, name='predictions')
    predictions = predictions.map(lambda x: 'Yes' if x == 1 else 'No')
    return predictions


@click.command("predict")
@click.option("--input-dir")
@click.option("--models-dir")
@click.option("--df")
@click.option("--predictions-dir")
def model_predict(input_dir, models_dir, df, predictions_dir):
    df = pd.read_csv(os.path.join(input_dir, df))
    config = read_training_pipeline_params("config.yaml")
    preds = infer_cv_model(df, os.path.join(models_dir, config.model_params.output_model_path))
    os.makedirs(predictions_dir, exist_ok=True)
    preds.to_csv(os.path.join(predictions_dir, "pred.csv"), index=False)


if __name__ == '__main__':
    model_predict()
