from datetime import datetime, timedelta
import json
import pandas
from airflow.operators import BashOperator
from airflow.models import DAG
import shutil
import sys
import subprocess

dag = DAG(dag_id='latent_factor_cf',
          schedule_interval='0 0 1 * *',
          dagrun_timeout=timedelta(hours=1),
          start_date=datetime(2014, 5, 1),
          end_date=datetime(2014, 7, 1)
          )

dag.catchup=True

def task(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    process_output = process.communicate()[0].strip()  # 0: output 1: error
    return process_output

t1 = \
    BashOperator(task_id='read_in_json',
                   bash_command=task('python ~/airflow/da\
                   gs/latent_factor_cf/read_in_json.py'),
                   dag=dag)

t2 = \
    BashOperator(task_id='SGD_CF_modeling',
                   bash_command=task('python ~/airflow/da\
                   gs/latent_factor_cf/SGD_CF_modeling.py'),
                   dag=dag)

t3 = \
    BashOperator(task_id='SGD_CF_validation',
                   bash_command=task('python ~/airflow/da\
                   gs/latent_factor_cf/SGD_CF_validation.py'),
                   dag=dag)

t4 = BashOperator(task_id='sleep',bash_command='sleep 6000', dag=dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
t4.set_upstream(t3)
