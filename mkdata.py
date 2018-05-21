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
trn_data = spt.split_normalized_xy(train)
test_data = spt.split_normalized_xy(test)
# split x, y
X_train, y_train = trn_data
X_test, y_test = test_data

#save normalized trainX, testX
import csv
with open('./data/preprocessed_train.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['label', 'review'])
    for i in range(len(X_train)):
        writer.writerow([y_train[i], X_train[i]])
        if i % 1000 == 0:
            print("{}th data processed!!".format(i))

with open('./data/preprocessed_test.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['label', 'review'])
    for i in range(len(X_test)):
        writer.writerow([y_test[i], X_test[i]])
        if i % 1000 == 0:
            print("{}th data processed!!".format(i))

'''
trn = pd.read_csv('./data/preprocessed_train.csv', header=0)
X_train, y_train= spt.split_xy(trn)
test = pd.read_csv('./data/preprocessed_test.csv', header=0)
X_test, y_test= spt.split_xy(test)
'''
