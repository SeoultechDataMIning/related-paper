import split as spt
import preprocessing as prep

import os
import pandas as pd

# txt to csv
if os.path.isfile("./data/test.ft.csv") == False:
    spt.make_csv("test.ft")
if os.path.isfile("./data/train.ft.csv") == False:
    spt.make_csv("train.ft")

#csv index:  label     review
train = pd.read_csv('./data/train.ft.csv', header=0)

row, col = train.shape
trnX = []
trnY = []
for i in range(row):
    # y만들기
    tmp_label = train['label'][i][:]
    if tmp_label == '__label__2':
        trnY.append(2)
    elif tmp_label == '__label__1':
        trnY.append(1)
    # x만들기
    corpus = train['review'][i][:]
    norm_corpus = prep.normalizer(corpus)
    trnX.append(norm_corpus)
    if i % 5000 == 0:
        print("{}번째 전처리".format(i))

print(trnX)
print(trnY)


