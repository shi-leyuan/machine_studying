import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.io as sio

#                    线性不可分svm

path = "E:/BaiduNetdiskDownload/data_sets/ex6data2.mat"
data = sio.loadmat(path)