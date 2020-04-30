# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:13:50 2020

@author: rmalves
"""

import os
import implicit
import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sparse
import pickle


os.chdir('c:\\Users\\rmalves\Desktop\RecSys')

df = pd.read_csv('candy.csv')
dfrename=df.rename(columns ={'item':'itemID',
                       'user':'userID',
                       'review':'rating'})

df_order = dfrename[['userID', 'itemID', 'rating']]


df_order['user_ID'] = df_order['userID'].astype("category").cat.codes
df_order['item_ID'] = df_order['itemID'].astype("category").cat.codes


data = df_order.drop(['itemID', 'userID'], axis=1)

# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user) 
# and one for recommendations (user-item)

sparse_item_user = sparse.csr_matrix((data['rating'].astype(float), (data['item_ID'], data['user_ID'])))
sparse_user_item = sparse.csr_matrix((data['rating'].astype(float), (data['user_ID'], data['item_ID'])))

# Initialize the als model and fit it using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (sparse_item_user * alpha_val).astype('double')

# Fit the model
model.fit(data_conf)

model.similar_items(1)[0][1]

with open('model.pkl','wb') as f:
     pickle.dump(model,f)


model.predict([3])


#http://127.0.0.1:8000/

"""
Created on Mon Apr 27 17:46:20 2020

@author: rmalves
"""
# import os
# import dill
# from surprise import NormalPredictor
# from surprise import SVD
# from surprise import Dataset
# from surprise import Reader
# from surprise import accuracy
# from surprise.model_selection import train_test_split
# import pandas as pd
# from surprise import KNNBasic


# os.chdir('c:\\Users\\rmalves\Desktop\RecSys')

# df = pd.read_csv('candy.csv')
# dfrename=df.rename(columns ={'item':'itemID',
#                        'user':'userID',
#                        'review':'rating'})

# df_order = dfrename[['userID', 'itemID', 'rating']]

# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df_order[['userID', 'itemID', 'rating']], reader)


# # sample random trainset and testset
# # test set is made of 25% of the ratings.
# trainset, testset = train_test_split(data, test_size=.01)

# # We'll use the famous SVD algorithm.
# model = KNNBasic()

# # Train the algorithm on the trainset, and predict ratings for the testset
# model.fit(trainset)

# sismMatrix=model.compute_similarities()

# testItemInnerID=trainset.to_inner_uid()

# testItemInnerID=trainset.to_inner_iid(1)

# iterator = data.all_ratings()
# new_df = pd.DataFrame(columns=['uid', 'iid', 'rating'])
# i = 0
# for (uid, iid, rating) in iterator:
#     new_df.loc[i] = [uid, iid, rating]
#     i = i+1

# new_df.head(2)


# # Then compute RMSE

# predictions = model.test(testset)
# accuracy.rmse(predictions)

# with open('model.pkl','wb') as f:
#     dill.dump(model,f)
    
# del model
# with open('model.pkl','rb') as f:
#           model = dill.load(f)
          
