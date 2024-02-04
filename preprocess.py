#import kmeans
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split




# reading the database from base folder base/Buy.csv
buy = pd.read_csv('base/Buy.csv')
abt = pd.read_csv('base/Abt.csv', encoding_errors='ignore')

# remove columns that are not needed
buy.drop(['id', 'manufacturer', 'price'], axis=1, inplace=True)
abt.drop(['id', 'price'], axis=1, inplace=True)

# convert the all to lower case
buy = buy.apply(lambda x: x.astype(str).str.lower())
abt = abt.apply(lambda x: x.astype(str).str.lower())

# taking just 500 items to test using train_test_split
trainBuy, testBuy = train_test_split(buy['name'], test_size=0.7, random_state=1)
trainAbt, testAbt = train_test_split(abt['name'], test_size=0.7, random_state=1)

# padronize the size of the data
trainBuy = trainBuy[:1]
trainAbt = trainAbt[:1]

# segment the data, first vectorize the data
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
vectorizer = CountVectorizer()
buyVectors = vectorizer.fit_transform(trainBuy)
abtVectors = vectorizer.fit_transform(trainAbt)
print(str(trainBuy))
print(str(trainAbt))
# cosine similarity of the common items
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
cosine_sim = cosine_similarity(buyVectors, abtVectors)
print(cosine_sim)
for i in range(len(cosine_sim)):
    if(cosine_sim[i][i] > 0.5):
        print(buy[i], abt[i])
        break

#print(buy.head())
#print(abt.head())

