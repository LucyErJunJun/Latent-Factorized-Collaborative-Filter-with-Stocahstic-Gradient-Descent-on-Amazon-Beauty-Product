import copy
import os
import pandas as pd
import math
import numpy as np
import json
from sklearn.metrics import mean_squared_error
import glob
import datetime,time

start_time=datetime.datetime.now()
start_time_str=datetime.datetime.strftime(start_time,'%Y-%m-%dT%H:%M:%S')
null=None
def parse(path):
    g = open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def data_raw_data(path1='/home/ec2-user/airflow/dags/latent_factor_cf/raw_data_past/*.json',path2='/home/ec2-user/airflow/dags/latent_factor_cf/raw_data_new/*.json'):
    # path1='./raw1/*.json'
	path_1=sorted(glob.glob(path1))
	path_2=sorted(glob.glob(path2))
	if len(path_1)==0:
		DF=getDF(path_2[0])
	elif len(path_1)>0:
		input_concat=[]
		for v in path_1:
			input_concat.append(getDF(v))
		if len(path_2)==0:
			print ("No New Input Available. Re-Train Model Uses Same Data!")
		else:
			input_concat.append(getDF(path_2[0]))
		DF=pd.concat(input_concat,ignore_index=True)
	f=open('/home/ec2-user/airflow/dags/latent_factor_cf/logs/logs.json','w')
	if len(path_2)>0:	
		f.write(json.dumps({start_time_str:[path_2[0]]}))	
	else:
		f.write(json.dumps({start_time_str:path_2}))
	f.close()
	return DF

df=data_raw_data()

df['unixReviewTime']=pd.to_datetime(df['unixReviewTime'],unit='s')
df.drop(['reviewTime'],axis=1,inplace=True)
# set unique ID for each review
df['ReviewID']=df.index+1
print (df.shape)
n_users = df.reviewerID.unique().shape[0]
n_items = df.asin.unique().shape[0]
print ('user: %d, item: %d' %(n_users,n_items))

df = df.sort_values(['reviewerID', 'asin'], ascending = [True, True])
df.asin=pd.core.categorical.Categorical(df.asin)
df['asin_id']=df.asin.cat.codes
df.reviewerID=pd.core.categorical.Categorical(df.reviewerID)
df['reviewer_ID']=df.reviewerID.cat.codes

df_ratings=df[['reviewer_ID','asin_id','overall']]
ratings = np.zeros((n_users, n_items))

for row in df_ratings.itertuples():
    ratings[row[1], row[2]] = row[3]

def apply_Ncore(ratings, N_core):
    n_users = ratings.shape[0]
    n_items = ratings.shape[1]
    items_id = [x for x in range(n_items) if len(ratings[:, x].nonzero()[0]) >= N_core]
    ratings = ratings[:, items_id]
    # map the new items position to the old
    item_link=list(zip(items_id,range(ratings.shape[1])))
    users_id = [x for x in range(n_users) if len(ratings[x, :].nonzero()[0]) >= N_core]
    # map the new user posistion to the old
    ratings = ratings[users_id, :]
    user_link=list(zip(users_id,range(ratings.shape[0])))
    return ratings,item_link,user_link

ratings_Ncore,items_link_Ncore,users_link_Ncore = apply_Ncore(ratings, 8)
print("After applying the 10-core, there are %d users and %d left." \
	%(ratings_Ncore.shape[0],ratings_Ncore.shape[1]))

# export map between Ncore index and ID
df['reviewerID_Ncore']=df['reviewer_ID'].map(dict(users_link_Ncore))
df['asinID_Ncore']=df['asin_id'].map(dict(items_link_Ncore))
item_map_afterNcore=df[['asinID_Ncore','asin']].drop_duplicates(keep='first')
user_map_afterNcore=df[['reviewerID_Ncore','reviewerID']].drop_duplicates(keep='first')

# output to json
f1=open('/home/ec2-user/airflow/dags/latent_factor_cf/logs/item_map_afterNcore_'+start_time_str+'.json','w')
f1.write(json.dumps(item_map_afterNcore.to_dict(orient='records')))
f1.close()

f2=open('/home/ec2-user/airflow/dags/latent_factor_cf/logs/user_map_afterNcore_'+start_time_str+'.json','w')
f2.write(json.dumps(user_map_afterNcore.to_dict(orient='records')))
f2.close()

# split into training and validation
def train_test_split_userwise(ratings,num_test=5):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    nonzero_id=np.transpose(np.array([ratings.nonzero()[0].tolist(),ratings.nonzero()[1].tolist()]))
    index=pd.DataFrame(nonzero_id,columns=['row','col'])
    gps =index.groupby(['row'])
    randx = lambda obj: obj.loc[np.random.choice(obj.index, num_test, False),:]
    test_ratings=gps.apply(randx).values.transpose()
    test_ratings_row=test_ratings[0].tolist()
    test_ratings_col=test_ratings[1].tolist()
    train[test_ratings_row,test_ratings_col] = 0.
    test[test_ratings_row,test_ratings_col] = ratings[test_ratings_row,test_ratings_col]   
    assert(np.all((train * test) == 0)) 
    return train, test

train_Ncore_user, test_Ncore_user = train_test_split_userwise(ratings_Ncore,num_test=3)
# check whether the test dataset has the same shape as the training
print('training dimension: ', train_Ncore_user.shape)
print('test dimension: ',test_Ncore_user.shape)

def normalize_userwise(train,test):
    user=0
    train_new=copy.deepcopy(train)
    test_new=copy.deepcopy(test)
    for i in range(train_new.shape[0]):
        items=np.nonzero(train_new[i,:])[0].tolist()
        user_avg=np.sum(train_new[i,items])/len(items)
        items_test=np.nonzero(test_new[i,:])[0].tolist()
        train_new[i,items]=(train_new[i,items]-user_avg)
        test_new[i,items_test]=(test_new[i,items_test]-user_avg)
    return train_new, test_new

train_Ncore_user_normalized,test_Ncore_user_normalized=normalize_userwise(train_Ncore_user,test_Ncore_user)
user_baseline_normalized=np.sum(test_Ncore_user_normalized**2)/len(np.nonzero(test_Ncore_user)[0])

# write the baseline MSE of the test set out for validation
MSE={}
MSE["Baseline"]=user_baseline_normalized
f3=open('/home/ec2-user/airflow/dags/latent_factor_cf/logs/MeanSquareError_'+start_time_str+'.json','w')
f3.write(json.dumps(MSE))
f3.close()

# write out the training and test set
np.savez_compressed('/home/ec2-user/airflow/dags/latent_factor_cf/training_matrix/training_'+start_time_str,train_Ncore_user)
np.savez_compressed('/home/ec2-user/airflow/dags/latent_factor_cf/test_matrix/test_'+start_time_str,test_Ncore_user)


