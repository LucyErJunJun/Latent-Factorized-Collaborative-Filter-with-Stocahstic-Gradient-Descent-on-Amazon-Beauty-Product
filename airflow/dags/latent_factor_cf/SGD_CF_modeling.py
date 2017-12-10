import numpy as np
import json
import glob
from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    pred = pred[actual.nonzero()[0].tolist(),actual.nonzero()[1].tolist()].flatten()
    actual = actual[actual.nonzero()[0].tolist(),actual.nonzero()[1].tolist()].flatten()
    mse = mean_squared_error(pred, actual)
    return mse

class RecommendationSGD_Random():
    
    def __init__(self, 
                 ratings, 
                 n_factors = 10, 
                 item_reg = 1.0, 
                 user_reg = 1.0,
                 item_bias_reg = 1.0,
                 user_bias_reg = 1.0,
                 max_iter = 15,
                 batch_size=3,
                 learning_rate = 0.01,
                 tolerance=0.0001,
                 verbose = True):
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning_rate = learning_rate
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)
        self.batch_size=batch_size
        self._v = verbose
        self.n_iter = max_iter
        self.MSE=[]
        self.tolerance=0-tolerance

    
    def fit(self):
        """ 
        Train model
        """       
        self.ratings_zero=np.zeros(self.ratings.shape)
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))
        
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        
        ctr = 1
        while ctr <= self.n_iter:
            if ctr % 10 == 0 :
                print ('\tcurrent iteration: {}'.format(ctr))
            
            if ctr>1:
                # predict the ratings by applying the vectorized prediction forluma
                ratings_pred=self.user_bias[:,np.newaxis]+self.item_bias[np.newaxis,:]+self.global_bias+self.user_vecs.dot(self.item_vecs.T)
                ratings_pred=np.nan_to_num(ratings_pred)
                
                self.MSE.append(get_mse(ratings_pred, self.ratings))
                if self._v:
                    print (self.MSE[-1])
            if len(self.MSE)>1:
                # set the tolerance on the difference in MSE, cut-off the iteration when the difference is less than tolerance
                if self.MSE[-1]<self.MSE[-2] and (self.MSE[-1]-self.MSE[-2])/self.MSE[-2]>self.tolerance:
                    return (self.user_vecs, self.item_vecs, self.user_bias, self.item_bias, self.global_bias)
            # one sample SGD
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            
            for start_idx in range(0, self.n_samples - self.batch_size + 1, self.batch_size):
                idx = self.training_indices[start_idx:start_idx + self.batch_size]
                u = self.sample_row[idx]
                i = self.sample_col[idx]
                
                # error
                e = [self.ratings[a,b] - self.predict(a,b) for a,b in zip(u,i)]
                
                # update biases
                self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])
                
                # update latent factors
                self.user_vecs[u, :] = [self.user_vecs[u, :][x] + self.learning_rate * (e[x] * self.item_vecs[i, :][x] - self.user_reg * self.user_vecs[u,:][x]) for x in range(self.batch_size)]          
                self.item_vecs[i, :] = [self.item_vecs[i, :][x] + self.learning_rate * (e[x] * self.user_vecs[u, :][x] - self.item_reg * self.item_vecs[i,:][x]) for x in range(self.batch_size)]
                
            ctr += 1
        
        
        return (self.user_vecs, self.item_vecs, self.user_bias, self.item_bias, self.global_bias)
    
    def predict(self, u, i):
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        prediction=np.nan_to_num(prediction)
        return prediction
    
    def get_prediction(self):
        vecs = self.fit()
        user_vecs = vecs[0]
        item_vecs = vecs[1]
        user_bias = vecs[2]
        item_bias = vecs[3]
        global_bias = vecs[4]
        predictions=user_bias[:,np.newaxis]+item_bias[np.newaxis,:]+global_bias+user_vecs.dot(item_vecs.T)
        return predictions

if __name__ == '__main__':
    train_path=sorted(glob.glob('/home/ec2-user/airflow/dags/latent_factor_cf/training_matrix/training_*.npz'))[-1]
    training=np.load(train_path)
    train_Ncore_user=training['arr_0']
    SGD_Pred_Matrix=RecommendationSGD_Random(train_Ncore_user,n_factors = 5,max_iter= 20,
                     batch_size=50,verbose=False).get_prediction()

    f=open('/home/ec2-user/airflow/dags/latent_factor_cf/logs/logs.json','r')
    time_set=json.load(f)
    f.close()
    for k,v in time_set.items():
        start_time_str=k
    np.savez_compressed('/home/ec2-user/airflow/dags/latent_factor_cf/logs/Pred_Matrix_'+start_time_str,SGD_Pred_Matrix)