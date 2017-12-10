import pandas as pd
import numpy as np
import json

class Recommender():
	def __init__(self,Pred,meta,user_map_afterNcore,item_map_afterNcore,N):
		self.Pred=Pred
		self.meta=meta
		self.user_map_afterNcore=user_map_afterNcore
		self.item_map_afterNcore=item_map_afterNcore
		self.N=N

	def recommend(self,user_id):
		user_in_list=self.user_map_afterNcore.ix[self.user_map_afterNcore['reviewerID']==user_id,'reviewerID_Ncore']
		if pd.notnull(user_in_list.values[0]):
			item_Ncore=np.argsort(self.Pred[int(user_in_list.values[0]),:])[:-self.N-1:-1].tolist()
			Recommend=self.item_map_afterNcore.loc[self.item_map_afterNcore['asinID_Ncore'].isin(item_Ncore),'asin'].tolist()
			dict_return={}
			dict_return['items']=self.meta.loc[self.meta['asin'].isin(Recommend),['asin','title','type_L1','type_L2','type_L3','description','imUrl']].to_dict('records')
			dict_return['user_id']=user_id
			return dict_return
		else:
			return {'user_id':user_id,'items':[]}

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

if __name__ == "__main__":
	pred = np.load('../airflow/dags/latent_factor_cf/logs/Pred_Matrix_2017-12-01T01:04:56.npz')
	trained_model_purged=pred['arr_0']
	print (trained_model_purged)
	meta = getDF('../models/meta_Beauty.json')
	for i in range(4):
		meta['type_L'+str(i)] = meta['categories'].apply(lambda x: x[0][i] if i<len(x[0]) else np.nan)
	f1=open('../airflow/dags/latent_factor_cf/logs/user_map_afterNcore_2017-12-01T01:04:56','r')
	user_id_map_dict=json.load(f1)
	user_id_map=pd.DataFrame(user_id_map_dict)
	f2=open('../airflow/dags/latent_factor_cf/logs/item_map_afterNcore_2017-12-01T01:04:56.json','r')
	item_id_map_dict=json.load(f2)
	item_id_map=pd.DataFrame(item_id_map_dict)
	intance=Recommender(Pred=trained_model_purged,meta=meta,user_map_afterNcore=user_id_map,item_map_afterNcore=item_id_map,N=10)
	print (intance.recommend(user_id="AZXP46IB63PU8"))