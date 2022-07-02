import os
import click
import pandas as pd
from sklearn.metrics import accuracy_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--models-dir")
def model_validate(input_dir: str, models_dir: str):
    true_labels = pd.read_csv(os.path.join(input_dir, "y_val.csv")).HeartDisease
    preds = pd.read_csv(os.path.join(input_dir, "pred.csv")).predictions
    preds = preds.map(lambda x: x == 'Yes')
    with open(os.path.join(models_dir, "metrics.txt"), "w") as f_out:
        f_out.write(f"Accuracy score on validation is {accuracy_score(true_labels.values, preds.values)}")


if __name__ == '__main__':
    model_validate()
