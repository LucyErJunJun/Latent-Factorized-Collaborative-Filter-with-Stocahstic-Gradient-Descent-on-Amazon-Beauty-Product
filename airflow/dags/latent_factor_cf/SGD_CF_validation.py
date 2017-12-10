import numpy as np
import json
from sklearn.metrics import mean_squared_error
import glob
import shutil
import re

def get_mse(pred, actual):
    pred = pred[actual.nonzero()[0].tolist(),actual.nonzero()[1].tolist()].flatten()
    actual = actual[actual.nonzero()[0].tolist(),actual.nonzero()[1].tolist()].flatten()
    mse = mean_squared_error(pred, actual)
    return mse

def path_to_matrix(path_str):
	path=sorted(glob.glob(path_str))[-1]
	path_np=np.load(path)
	return path_np['arr_0']

if __name__ == '__main__':
	MSE_path=sorted(glob.glob('/home/ec2-user/airflow/dags/latent_factor_cf/logs/MeanSquareError_*.json'))[-1]
	f=open(MSE_path,'r')
	MSE=json.load(f)
	f.close()
	test_Ncore_user=path_to_matrix('/home/ec2-user/airflow/dags/latent_factor_cf/test_matrix/test_*.npz')
	pred_Ncore_user=path_to_matrix('/home/ec2-user/airflow/dags/latent_factor_cf/logs/Pred_Matrix_*.npz')
	MSE['Validation']=get_mse(pred_Ncore_user,test_Ncore_user)
	f=open(MSE_path,'w')
	f.write(json.dumps(MSE))
	f.close()
	f1=open('/home/ec2-user/airflow/dags/latent_factor_cf/logs/logs.json','r')
	items=json.load(f1)
	for k,v in items.items():
		if len(v)==0:
			print ("No New Input To Move")
		else:
			m=re.match('.*/(.*.json)',v[0])
			shutil.move(v[0],'/home/ec2-user/airflow/dags/latent_factor_cf/raw_data_past/'+m.group(1))



