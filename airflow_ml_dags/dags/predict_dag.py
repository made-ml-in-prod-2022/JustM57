from datetime import timedelta
from airflow import DAG
from pendulum import today
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.models import Variable


MOUNT_DIR = "/home/dmitry/Documents/made/ml_in_prod_2022/JustM57/airflow_ml_dags/data"
INPUT_DIR = "/data/raw/{{ ds }}"
PROCESSED_DIR = "/data/processed/{{ ds }}"
MODELS_DIR = Variable.get("model_dir")
PREDICTION_DIR = "/data/predictions/{{ ds }}"


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=today('UTC').add(days=-7)
) as dag:
    preprocess = DockerOperator(
        image="airflow-test-preprocess",
        command=f"--input-dir {INPUT_DIR} --output-dir {PROCESSED_DIR} --models-dir {MODELS_DIR}",
        task_id="docker-airflow-test-process",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {PROCESSED_DIR} --models-dir {MODELS_DIR} --df data.csv --predictions-dir {PREDICTION_DIR}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    preprocess >> predict
