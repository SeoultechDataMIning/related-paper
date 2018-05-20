import pickle
import pandas as pd
import split as spt

filename = open('./model_save/naive_bayes.pkl', 'r+b')
clf = pickle.load(filename)

test = pd.read_csv('./data/test.ft.csv', header=0)
test_data = spt.split_xy(test)
X_test, y_test = test_data
print(clf.score(X_test, y_test))
