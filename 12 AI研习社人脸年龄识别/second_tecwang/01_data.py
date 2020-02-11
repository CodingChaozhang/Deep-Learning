import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from libs.utils import extract_fname


# # file_dir = "../../data/train/"
# # dst_filename = "train"
# file_dir = "../../data/test/"
# dst_filename = "test"
#
# image_size = 200
# dirs = os.listdir(file_dir)
#
# face_data = None
# for dir in dirs:
#     # dir can be seen as label
#     print(dir)
#
#     if os.path.isdir(os.path.join(file_dir, dir)):
#         # 列出该年龄段下所有图像
#         fs = os.listdir(os.path.join(file_dir, dir))
#
#         fs.sort(key= lambda x: extract_fname(x))
#
#         # 此处需要无符号数的uint8，而不能使用int8，符号位会导致负号产生
#         temp_specific_age = np.zeros((len(fs), image_size, image_size, 3), dtype="uint8")    # 预定义存储变量
#         # 如果是训练集，要生成label数据
#         temp_specific_age_label = None
#         if dst_filename == "train":
#
#             temp_specific_age_label = np.ones((len(fs), 1), dtype="uint8")
#             temp_specific_age_label = temp_specific_age_label * int(dir)
#             print(temp_specific_age_label[0])
#
#         for f_index in range(len(fs)):
#             suffix = os.path.splitext(fs[f_index])[1]
#             if suffix in [".png"]:
#                 img = plt.imread(os.path.join(file_dir, dir, fs[f_index]))
#                 # print(np.max(img), np.min(img))
#                 img = img*255
#                 img = img.astype("uint8")
#                 # print(np.max(img), np.min(img))
#                 temp_specific_age[f_index] = img
#
#                 # 预览图片是否正确
#                 # plt.imshow(temp_specific_age[f_index])
#                 # plt.show()
#             else:
#                 logging.warn("Unexpected format file!")
#
#         if face_data is None:
#             print("Initialization the face_data data")
#             face_data = temp_specific_age.copy()
#             print("face_data.shape", face_data.shape)
#
#             if dst_filename == "train":
#                 face_label = temp_specific_age_label.copy()
#                 print("face_label.shape", face_label.shape)
#         else:
#             print("Add more label data to face_data")
#             face_data = np.vstack((face_data, temp_specific_age))
#             print("face_data.shape", face_data.shape)
#
#             if dst_filename == "train":
#                 face_label = np.vstack((face_label, temp_specific_age_label))
#                 print("face_label.shape", face_label.shape)
#
#
# # plt.subplot(1,3,1)
# # plt.imshow(face_data[0])
# # plt.subplot(1,3,2)
# # plt.imshow(face_data[1])
# # plt.subplot(1,3,3)
# # plt.imshow(face_data[2])
# # plt.show()
#
# np.save(dst_filename + ".npy", face_data)
# print("final face_data shape", face_data.shape)
# # if dst_filename == "train":
# #     # save the lable data to csv file
# #     print(face_label.shape)
# #     label = pd.DataFrame(face_label)
# #     label.to_csv(dst_filename + "_label.csv")
# #     # np.save(dst_filename + "_label.npy", face_label)
# #     print("final label shape", label.shape)
# #     print(label.head())

def solve_test():
    file_dir = "../../data/"
    dir = 'test'
    dst_filename = "test"
    image_size = 200
    face_data = None
    fs = os.listdir(os.path.join(file_dir, 'test'))
    fs.sort(key=lambda x: extract_fname(x))

    # 此处需要无符号数的uint8，而不能使用int8，符号位会导致负号产生
    temp_specific_age = np.zeros((len(fs), image_size, image_size, 3), dtype="uint8")  # 预定义存储变量

    for f_index in range(len(fs)):
        suffix = os.path.splitext(fs[f_index])[1]
        if suffix in [".png"]:
            img = plt.imread(os.path.join(file_dir, dir, fs[f_index]))
            img = img * 255
            img = img.astype("uint8")
            temp_specific_age[f_index] = img
        else:
            logging.warn("Unexpected format file!")

    if face_data is None:
        print("Initialization the face_data data")
        face_data = temp_specific_age.copy()
        print("face_data.shape", face_data.shape)
    else:
        print("Add more label data to face_data")
        face_data = np.vstack((face_data, temp_specific_age))
        print("face_data.shape", face_data.shape)

    np.save("data/test.npy", face_data)
    print("final face_data shape", face_data.shape)


if __name__ == '__main__':
    solve_test()
