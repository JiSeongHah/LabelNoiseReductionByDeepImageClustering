import csv
import pandas as pd
import os
from glob import glob


def save_important_data(path,name,loop_num,data_lst):
    with open(path+name+'_result'+'.csv','a',newline='') as f:
        wrtr = csv.writer(f)
        for i in range(len(data_lst[0])):
            wrtr.writerow([loop_num,i, data_lst[0][i], data_lst[1][i], data_lst[2][i], data_lst[3][i] ])

    print('saving rewards and accs done completely')


def lst2csv(save_dir,save_name,**kwargs):
    dict4df = dict()

    for kwarg in kwargs.items():
        name = kwarg[0]
        lst = kwarg[1]

        dict4df[str(name)] = lst

    data_df = pd.DataFrame(dict4df)

    data_df.to_csv(save_dir+save_name+'.csv',header=True,index=True)
    print('saving lst to csv complete')


def mk_name(*args,**name_value_dict):
    total_name = ''
    additional_arg = ''

    for arg in args:
        additional_arg += (str(arg)+'_')

    for name_value in name_value_dict.items():
        name = name_value[0]
        value = name_value[1]
        total_name += (str(name)+str(value)+'_')

    total_name += additional_arg[:-1]

    return total_name


# make directory.
# if directory doesn't exist, it makes.
# if directory exist already, it doens't make new one
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'making {str(directory)} complete successfully!')
    except OSError:
        print("Error: Failed to create the directory.")

def load_my_model(path):
    model_lst = glob(path+'*.pt')

    #print(model_lst)
    sorted_model_lst = sorted(model_lst,key=lambda x:  (x.split('.')[0]).split('/')[-1]  )[0]
    print('lst sored is : ',sorted_model_lst)
    return sorted_model_lst

# delete file which is not in list
def delFileOrFolders(dir,exceptionLst):

    fileLst = os.listdir(dir)

    for eachFile in fileLst:
        if eachFile not in exceptionLst:
            os.remove(dir+eachFile)
            print(f'{dir}+{eachFile} removed')

