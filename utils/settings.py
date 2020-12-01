import os
from os.path import abspath, dirname, join
import platform
import numpy as np


if os.name == 'nt':
    DATA_DIR = 'C:/Users/zfj/research-data/wechat-wow'
    OUT_DIR = 'C:/Users/zfj/research-out-data/wechat-wow'
else:
    if platform.system() == "Darwin":
        DATA_DIR = '/Users/zfj/research-data/wechat-wow'
        OUT_DIR = '/Users/zfj/research-out-data/wechat-wow'
    else:
        DATA_DIR = '/home/zfj/research-data/wechat-wow'
        OUT_DIR = '/home/zfj/research-out-data/wechat-wow'


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# TEST_SIZE = 10001
TEST_SIZE = np.iinfo(np.int64).max
