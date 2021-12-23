import csv

def lst2csv(lst,dir,name):
    with open(dir+name,'a',newline='') as f:
        write = csv.writer(f)
        write.writerows(lst)

    print('converting test lst to csv file complete')
    print(f'final file saved in {dir+name}')