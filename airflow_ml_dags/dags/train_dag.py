from datetime import timedelta
from airflow import DAG
from pendulum import today
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


MOUNT_DIR = "/home/dmitry/Documents/made/ml_in_prod_2022/JustM57/airflow_ml_dags/data"
INPUT_DIR = "/data/raw/{{ ds }}"
PROCESSED_DIR = "/data/processed/{{ ds }}"
MODELS_DIR = "/data/models/{{ ds }}"


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=today('UTC').add(days=-7)
) as dag:
    preprocess = DockerOperator(
        image="airflow-train-preprocess",
        command=f"--input-dir {INPUT_DIR} --output-dir {PROCESSED_DIR} --models-dir {MODELS_DIR}",
        task_id="docker-airflow-train-process",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {PROCESSED_DIR}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    fit = DockerOperator(
        image="airflow-fit",
        command=f"--input-dir {PROCESSED_DIR} --models-dir {MODELS_DIR}",
        task_id="docker-airflow-fit",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {PROCESSED_DIR} --models-dir {MODELS_DIR} --df x_val.csv --predictions-dir {PROCESSED_DIR}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {PROCESSED_DIR}  --models-dir {MODELS_DIR}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=MOUNT_DIR, target="/data", type='bind')]
    )

    preprocess >> split >> fit >> predict >> validate
