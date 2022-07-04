from datetime import timedelta
from airflow import DAG
from pendulum import today
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "create_fake_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=today('UTC').add(days=-14)
) as dag:
    fake = DockerOperator(
        image="fake_data",
        command="/data/raw/{{ ds }}",
        task_id="docker-airflow-fake",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/dmitry/Documents/made/ml_in_prod_2022/JustM57/airflow_ml_dags/data",
                      target="/data", type='bind')]
    )
