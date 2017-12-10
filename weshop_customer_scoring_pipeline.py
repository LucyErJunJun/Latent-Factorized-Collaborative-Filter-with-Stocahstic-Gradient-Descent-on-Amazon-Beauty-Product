from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

dag = DAG('weshop_customer_scoring_pipeline', description='datapipeline scores each weshop customer',
          schedule_interval='0 12 * * *',
          start_date=datetime(2017, 11, 19), catchup=False)


def ingest_data():
    return "data ingested"

ingest_data_operator = PythonOperator(task_id='ingest_data_task',
 python_callable=ingest_data, dag=dag)

def preprocess_data():
    return "data preprocessed"

preprocess_data_operator = PythonOperator(task_id='preprocess_data_task',
 python_callable=preprocess_data, dag=dag)


def generate_features():
    return "feature generation complete"

generate_features_operator = PythonOperator(task_id='generate_features_task',
 python_callable=generate_features, dag=dag)

def score_features():
    return "scoring features complete"

score_features_operator = PythonOperator(task_id='score_features_task',
 python_callable=score_features, dag=dag)

def publish_scores():
    return "scores published"

publish_scores_operator = PythonOperator(task_id='publish_scores_task',
 python_callable=publish_scores, dag=dag)