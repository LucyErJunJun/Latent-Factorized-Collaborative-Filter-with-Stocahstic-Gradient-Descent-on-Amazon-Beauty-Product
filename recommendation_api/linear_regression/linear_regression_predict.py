from sklearn.externals import joblib
import numpy as np



class LinearRegressionPredictor: 

    def __init__(self):
        pass
        
    def loadModel(self):
        regression_model = joblib.load('regression_model_v1.pkl') 
        return regression_model   

    def  predict(self, features): 
        regression_model = self.loadModel()
        prediction = regression_model.predict(features)
        # need to make the preciction json serializable
        pred = str(prediction)
        return pred
