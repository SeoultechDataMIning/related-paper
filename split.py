#파일은 같은 폴더에 있어야됨
import csv
def make_csv(filename):
    f = open("./data/" + filename + ".txt")
    f_csv = open("./data/" + filename + ".csv", 'w')
    count = 0
    csvWrite = csv.writer(f_csv)
    csvWrite.writerow(["label", "review"])

    while True:
        line = f.readline()
        if not line: break
        csvWrite.writerow([line[:10], line[11:]])
        count += 1
        print(str(count)+'line make')
    f.close()
    f_csv.close()



import preprocessing as prep
def split_xy(raw_data):
    row, col = raw_data.shape
    trnX = []
    trnY = []
    for i in range(row):
        # y만들기
        tmp_label = raw_data['label'][i][:]
        if tmp_label == '__label__2':
            trnY.append(1)
        elif tmp_label == '__label__1':
            trnY.append(0)
        # x만들기
        corpus = raw_data['review'][i][:]
        norm_corpus = prep.normalizer(corpus)
        trnX.append(norm_corpus)
        if i % 5000 == 0:
            print("{}번째 전처리".format(i))

    return (trnX, trnY)

