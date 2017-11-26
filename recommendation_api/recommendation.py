import pandas as pd
import numpy as np

class Recommender():
	def __init__(self,Pred=trained_model,meta=product,user_map_afterNcore=user_id_map,item_map_afterNcore=item_id_map,ratings_Ncore=ratings,N=number_of_items):
		self.Pred=Pred
		self.meta=meta
		self.user_map_afterNcore=user_map_afterNcore
		self.item_map_afterNcore=item_map_afterNcore
		sef.ratings_Ncore=ratings_Ncore
		self.N=N

	def recommend(user_id):
		user_in_list=self.user_map_afterNcore.ix[self.user_map_afterNcore['reviewerID']==user_id,'reviewerID_Ncore']
		if pd.notnull(user_in_list.values[0]):
	        #replace all the item rated by users as zero to avoid recommend item user already bought
	        self.Pred[np.where(self.ratings_Ncore!=0)[0],np.where(self.ratings_Ncore!=0)[1]]=0
	        #find what the user has rated
	        item_bought=np.nonzero(self.ratings_Ncore[int(user_in_list.values[0]),:])[0].tolist()
	        asin_bought=self.item_map_afterNcore.loc[self.item_map_afterNcore['asinID_Ncore'].isin(item_bought),'asin'].tolist()
	        #recommend items
	        item_Ncore=np.argsort(self.Pred[int(user_in_list.values[0]),:])[:-N-1:-1].tolist()
	        Recommend=self.item_map_afterNcore.loc[self.item_map_afterNcore['asinID_Ncore'].isin(item_Ncore),'asin'].tolist()
	        dict_return['items']=self.meta.loc[self.meta['asin'].isin(Recommend),['asin','title','type_L1','type_L2','type_L3','description','imUrl']].to_dict('records')
	        dict_return['user_id']=user_id
	        return dict_return
	    else:
	        return {'user_id':user_id,'items':[]}
