import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from tqdm import tqdm

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import *

width = 224
size = 224
batch_s = 32

# 文件功能: 直接加载模型验证和测试(并输出结果到csv文件),没有利用输出结果
# 未注释部分是一个图片一个图片进行测试,可以看到进度条
# 注释掉的部分是一次将全部图片读入内存,然后测试,没有进度条,所以感觉很慢...

#############################################################
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(width, width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predict = Dense(70, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predict)

model.load_weights('keras_models/densenet121_0.2560_smt0.448.hdf5')

#############################################################

# with open('dataset/val_imglist.txt') as f:
#     lines = f.readlines()
#     n_valid = len(lines)
#     print('num of valid:', n_valid)
#
#     num_correct = 0.0
#     for line in tqdm(lines):
#         img_path, age = line[:-1].split(' ')
#         img = (cv2.resize(cv2.imread('dataset/face_age/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
#         img = img.reshape(1, 224, 224, 3)
#         prediction = model.predict(img, batch_size=1)
#         pred = np.argmax(prediction)
#         num_correct += (pred == int(age))
#
#     print("val_acc: ", num_correct / n_valid)

#############################################################

with open('dataset/test_imglist.txt') as f:
    lines = f.readlines()
    n_test = len(lines)
    print('num of test:', n_test)
    output_txt = open('key.csv', 'a')

    for line in tqdm(lines):
        img_path = line.strip()
        img_name = line.strip().split('/')[1].split('.')[0]
        img = (cv2.resize(cv2.imread('data/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img, batch_size=1)
        pred = np.argmax(prediction)
        age = "%03d" % (int(pred) + 1)
        text = img_name + ',' + age + '\n'
        output_txt.write(text)
    output_txt.close()

print("all done")

# def load_val_data(txt_path, size=224):
#     with open(txt_path) as f:
#         lines = f.readlines()
#
#     n = len(lines)
#     x = np.zeros((n, size, size, 3), np.float32)
#     y = np.zeros((n, 70), np.float)
#
#     for i, line in enumerate(tqdm(lines)):
#         img_path, age = line[:-1].split(' ')
#         x[i] = (cv2.resize(cv2.imread('dataset/face_age/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
#         y[i][int(age)] = 1
#     return x, y
#
#
# def load_test_data(txt_path, size=224):
#     with open(txt_path) as f:
#         lines = f.readlines()
#
#     n = len(lines)
#     x = np.zeros((n, size, size, 3), np.float32)
#     img_name = []
#     for i, line in enumerate(tqdm(lines)):
#         img_path = line.strip()
#         x[i] = (cv2.resize(cv2.imread('dataset/face_age/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
#         img_name.append((line.strip().split('/')[1]).split('.')[0])
#     return x, img_name
#
#
# width = 224
# batch_s = 32
#
# #############################################################
# base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(width, width, 3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# predict = Dense(70, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predict)
#
# model.load_weights('keras_models/densenet121_0.2474_smt0.655.hdf5')
#
# #############################################################
# x_valid, y_valid = load_val_data('dataset/val_imglist.txt', width)
#
# n_valid = len(x_valid)
# print('num of valid:', n_valid)
#
# num_correct = 0.0
# predictions = model.predict(x_valid, batch_size=1)
# for i, prediction in tqdm(enumerate(predictions)):
#     pred = np.argmax(prediction)
#     true = np.argmax(y_valid[i])
#     num_correct += (pred == true)
#
# print("val_acc: ", num_correct / n_valid)
#
#
# #############################################################
# x_test, img_name = load_test_data('dataset/test_imglist.txt', width)
#
# n_test = len(x_test)
# print('num of test:', n_test)
#
# predictions = model.predict(x_test, batch_size=1)
# output_txt = open('keras_output.csv', 'a')
# for i, prediction in tqdm(enumerate(predictions)):
#     pred = np.argmax(prediction)
#     age = "%03d" % (int(pred) + 1)
#     text = img_name[i] + ',' + age + '\n'
#     output_txt.write(text)
# output_txt.close()
#
# print("all done")
