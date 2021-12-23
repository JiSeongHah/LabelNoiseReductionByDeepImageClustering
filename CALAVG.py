import numpy as np

def cal_avg_error(x,y):
    x = np.asarray(x)
    y = np.asarray(y)

    error = np.mean(abs(x-y))

    return error

