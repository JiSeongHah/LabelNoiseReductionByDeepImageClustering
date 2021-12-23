import numpy as np

def conv_test(test_lst,threshold,range_num):
    if np.mean(test_lst[-range_num:]) <=threshold:
        return True
    else:
        return False