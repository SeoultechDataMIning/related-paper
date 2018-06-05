import split as spt

import os
import pandas as pd

'''
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
'''
'''
train = pd.read_csv('./data/preprocessed_train.csv', header=0)
test = pd.read_csv('./data/preprocessed_test.csv', header=0)

trn_data = spt.split_xy(train)
test_data = spt.split_xy(test)

X_train, y_train = trn_data
X_test, y_test = test_data
'''

train = pd.read_csv('data/preprocessed_train.csv', names=['label', 'review'], header=0)
test = pd.read_csv('data/preprocessed_test.csv', names=['label', 'review'], header=0)

train = train.fillna(' ')
test = test.fillna(' ')

trn_data = spt.split_xy(train)
test_data = spt.split_xy(test)

X_train, y_train = trn_data
X_test, y_test = test_data
print('data loaded!!')

# bag of word
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

'''
pipe = Pipeline([('vect', CountVectorizer()),
                ('clf', MultinomialNB())])

param_grid = {'vect__min_df': [1, 2, 3, 4, 5],
        'clf__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
'''
vectorizer = CountVectorizer(min_df=5)
Xtrain = vectorizer.fit_transform(X_train)
Xtest = vectorizer.fit_trainsform(X_test)

filename = "./model_save/naive_bayes_"
import datetime
now = datetime.datetime.now().strftime("%Y%m%d%H%M")
filename = './model_save/naive_bayes_' + now

import csv
f = open(filename + ".csv", "w")
csvWrite = csv.writer(f)
csvWrite.writerow(["min_dif", "alpha", "score"])

count = 1
for alpha in param_grid['clf__alpha']:
    clf = MultinomialNB(alpha=alpha)
    clf.fit(Xtrain, y_train)
    print(str(count) + "/30 model trained")
    score = clf.score(Xtest, y_test)
    print(str(count) + "/30 model tested")
    csvWrite.writerow([str(min_df), str(alpha), str(score)])
    count = count + 1
f.close()
        

    

'''
#cv: the number of folds ( cross-validation )
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train) 

print("training time: {}s".format(time.time() - start_time))
print("best params: {}".format(grid.best_params_))
print("best cross validation score: {}".format(grid.best_score_))
print("test set score: {}".format(grid.score(X_test, y_test)))

# save model
import pickle
import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M")
filename = './model_save/naive_bayes_' + now + '.pkl'

output = open(filename, 'w+b')
pickle.dump(grid.best_estimator_, output)
output.close()
'''



