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


if __name__ == "__main__":
    res = add_prefix_to_result(69)
    print(res)
