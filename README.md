
# Intro

Adam: Help!!! I think I need to start to use more skin care products.

Me: Oh OK. Any produt that you are considering?

Adam: No... that's why I need your help. Can you just pick whatever for me?

Me: (roll my eyes)


This is a typical story of our (single) guy friends, who don't want to spend a single second on researching skin care products but hope that their female friends will know them well enough to make educated guess about what products they would love. We thought maybe one day we can leverage our cross-domain knowledge in beauty products and data science to build a recommendation system for them. So here came our pet project as a starting point -  practising building recommendation system using Amazon beauty products data. 

In this Amazon Beauty Products project, we constructed a recommendation system that makes recommendations of beauty products to Amazon users based on their purchase history. Here we will share our step-by-step script, and highlight the enhancement we made to improve our model. 

After building up the collaborative filter, we are going to push the model into production by leveraging Apache Airflow and Flask. To simulate the validation and re-caliberation processing in production, the raw data is splited into four datasets along with the timeline,re-trained and validated batchly in Aiflow, which saved under the folder _airflow_. After the predicted user-item matrix is validated in Airflow, we push it into the recommandation api saved under the foler _recommendation_api_ and use Flask to return the prediction for individual user.



# Algorithm

We used factor based method to build this collaborative filtering system. We trained the model using Alternating Least Squares and Stochastic Gradient Descent respectively, and evaluated the performance on the both by comparing with the baseline Mean Square Error. It turns out Stochastic Gradient Descent having the better model performance. We also improved the model from several aspects:

The original review dataset came with 5-core. We further reduced it to 8-core to reduce the unstability introduced by randomness at train-test split process.

Users' rating behaviour could vary a lot. Some users tend to give an average rating significantly higher than 3 while others might give an average significantly lower than 3. Realizing the bias on both user-wise and item-wise, we normalized the ratings at user level before training the  Alternating Least Squares Collaborative Filter. 

We also enhanced Stochastic Gradient Descent by doing the following:
1. Add user bias and item bias to the model;
2. Apply mini-batch to updating the gradient descent;
3. Add pre cut-off on the iteration when percentage change on the MSE between epoches is lower than the threshold.
<br/>

Finally, we built a recommendation system returns "real" items with details to a specific user based on the prediction to test whether the prediction makes sense in an intuitive way. The recommendation system recommends the top ranked N items to a specific user along with what category of products did the user rate previously.

# Production

# Resources

### Data:

Amazon's product meta data and review data are available to the public, and are excellent source for practising recommendation system. For our project, we sourced the [review data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz) and [meta data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz) of beauty products from [Julian McAuley](http://jmcauley.ucsd.edu/data/amazon/).

### Readings:

WeCloudData course material (this material is available to studets who enroll in WeCloudData's capstone course; please contact us for further information.)

## Requirements

1. [Python 3.5](https://www.python.org/downloads/)
2. [Jupyter Notebook](http://jupyter.org)
3. [sklearn](http://scikit-learn.org/stable/)
4. [seaborn](https://seaborn.pydata.org)
5. [matplotlib](http://matplotlib.org)
6. [NumPy](http://www.numpy.org)
7. [pandas](http://pandas.pydata.org)
