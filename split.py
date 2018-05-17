#파일은 같은 폴더에 있어야됨
import csv
def make_csv(filename):
    f = open("./data/" + filename + ".txt")
    f_csv = open("./data/" + filename + ".csv", 'w')
    count = 0
    csvWrite = csv.writer(f_csv)
    csvWrite.writerow(["label", "review"])

    #while True:
    for i in range(20):
        line = f.readline()
        if not line: break
        csvWrite.writerow([line[:10], line[11:]])
        count += 1
        print(str(count)+'line make')
    f.close()
    f_csv.close()

