#파일은 같은 폴더에 있어야됨
import csv
f = open("./train.ft.txt")
f_csv = open("./train.csv", 'w')
count = 0
while True:
    line = f.readline()
    if not line: break
    csvWrite = csv.writer(f_csv)
    csvWrite.writerow([line[:10], line[11:]])
    count += 1
    print(str(count)+'line make')
f.close()
f_csv.close()