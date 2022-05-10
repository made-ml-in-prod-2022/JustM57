from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="heart_desease_classification",
    packages=find_packages(),
    version="0.0.1",
    description="heart_desease_classification",
    author="Dmitry Torshin",
    entry_points={
            "console_scripts": [
                "ml_example_train = src.train_pipeline:train_pipeline_wrapper"
            ]
        },
    install_requires=required,
    license="MIT",
)