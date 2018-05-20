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
X_train, y_train = trn_data
X_test, y_test = test_data



# bag of word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()

pipe = Pipeline([('vect', CountVectorizer()),
                ('clf', MultinomialNB())])
param_grid = {'vect__min_df': [1, 2, 3, 4, 5],
        'clf__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}

#cv: the number of folds ( cross-validation )
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train) 

print("training time: {}s".format(time.time() - start_time))
print("best params: {}".format(grid.best_params_))
print("best cross validation score: {}".format(grid.best_score_))
print("test set score: {}".format(grid.score(X_test, y_test)))

# save model
import pickle
output = open('./model_save/naive_bayes.pkl', 'w+b')
pickle.dump(grid.best_estimator_, output)
output.close()



