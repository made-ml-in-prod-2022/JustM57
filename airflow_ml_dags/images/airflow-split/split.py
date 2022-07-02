import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.2


@click.command("make_split")
@click.option("--input-dir")
def make_split(input_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=TEST_SIZE)
    x_train.to_csv(os.path.join(input_dir, "x_train.csv"), index=False)
    x_val.to_csv(os.path.join(input_dir, "x_val.csv"), index=False)
    y_train.to_csv(os.path.join(input_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(input_dir, "y_val.csv"), index=False)


if __name__ == "__main__":
    make_split()
