import keras
import numpy as np
from libs.network import *
import time

model = keras.models.load_model("data/model/model2019-11-01-11-15-53.h5", custom_objects={
    "classification_loss_stage2": classification_loss_stage2,
    "square_loss_tec": square_loss_tec,
    "cross_loss_tec": cross_loss_tec})

x_test = np.load("./test.npy")
x_test = x_test / 255.0

y_pred = model.predict(x_test)
print(y_pred)
y_pred_ages = np.argmax(y_pred, axis=1)
print(y_pred_ages)

import numpy as np
import re
import os


def add_prefix_to_result(res, len_result=3):
    res += 1
    res_arr = re.findall("\d", str(res))
    res_arr.reverse()

    out = np.zeros((len_result,), dtype="uint8")  # 初始化输出数组
    for d_index in range(len(res_arr)):
        out[out.shape[0] - d_index - 1] = int(res_arr[d_index])

    out = out.tolist()
    out = "".join(str(x) for x in out)

    return out


def extract_fname(x):
    num = int(os.path.splitext(x)[0])
    return num


index = 1
f = open("submission.csv", "w+")
for age in y_pred_ages:
    f.write(str(index) + "," + add_prefix_to_result(age) + "\n")
    index += 1

f.close()
