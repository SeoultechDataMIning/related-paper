import split as spt

import os
import pandas as pd

# txt to csv
if os.path.isfile("./data/test.ft.csv") == False:
    spt.make_csv("test.ft")
if os.path.isfile("./data/train.ft.csv") == False:
    spt.make_csv("train.ft")

#csv index:  label     review
train = pd.read_csv('./data/train.ft.csv', header=0)
test = pd.read_csv('./data/test.ft.csv', header=0)

# excute preprocessing in split_xy()
trn_data = spt.split_xy(train)
test_data = spt.split_xy(test)
# split x, y
trnX, y_train = trn_data
testX, y_test = test_data



# bag of word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
 # it might be better to try with min_df=5 ,if we have enough time.
vect = CountVectorizer(min_df=2).fit(trnX)
X_train = vect.transform(trnX)
X_test = vect.transform(testX)

'''
 X_train: train 데이터의 bag of word
 X_test: test 데이터의 bag of word
 y_train: train 데이터의 class
 y_test: test 데이터의 class
'''


