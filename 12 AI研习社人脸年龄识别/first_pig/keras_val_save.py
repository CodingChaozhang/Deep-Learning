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

# 文件功能: 输出每个模型的运行结果(每个类别的概率值),验证集和测试集都输出,每个模型保存成一个同名的txt文件


# 读取验证集val数据,返回读取的数据和onehot编码
# (由于这里只是进行验证,模型只需要数据并不需要标签,标签只是用来验证准确率的,所以统一用onehot编码就可以了)
def load_val_data(txt_path, size=224):
    with open(txt_path) as f:
        lines = f.readlines()

    n = len(lines)
    x = np.zeros((n, size, size, 3), np.float32)
    y = np.zeros((n, 70), np.float)

    for i, line in enumerate(tqdm(lines)):
        img_path, age = line[:-1].split(' ')
        x[i] = (cv2.resize(cv2.imread('dataset/face_age/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
        y[i][int(age)] = 1
    return x, y


# 读取测试集test数据并返回
def load_test_data(txt_path, size=224):
    with open(txt_path) as f:
        lines = f.readlines()

    n = len(lines)
    x = np.zeros((n, size, size, 3), np.float32)
    for i, line in enumerate(tqdm(lines)):
        img_path = line.strip()
        x[i] = (cv2.resize(cv2.imread('dataset/face_age/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
    return x


# 输入图片的大小
width = 224


#############################################################
# 定义模型,分别使用预训练模型densenet121和resnet50
base_densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(width, width, 3))
xd = base_densenet.output
xd = GlobalAveragePooling2D()(xd)
xd = Dense(512, activation='relu')(xd)
predictd = Dense(70, activation='softmax')(xd)

base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(width, width, 3))
xr = base_resnet.output
xr = GlobalAveragePooling2D()(xr)
xr = Dense(512, activation='relu')(xr)
predictr = Dense(70, activation='softmax')(xr)

model_list = []
# 添加需要输出结果的模型(单独输出)
model_list.append('densenet121_0.2397_baseline.hdf5')
model_list.append('densenet121_0.2455_smt0.8.hdf5')
model_list.append('densenet121_0.2474_smt0.655.hdf5')
model_list.append('densenet121_0.2569_3.7619_0.517.hdf5')
model_list.append('densenet121_0.2560_smt0.448.hdf5')

model_dir = "keras_models/"
# 验证集val输出阶段
for i, model in enumerate(model_list):
    if model[0] == 'd':
        model = Model(inputs=base_densenet.input, outputs=predictd)
    elif model[0] == 'r':
        model = Model(inputs=base_resnet.input, outputs=predictr)
    model.load_weights(model_dir + str(model_list[i]))

    x_valid, y_valid = load_val_data('dataset/val_imglist.txt', width)

    n_valid = len(x_valid)
    print('num of valid:', n_valid)

    num_correct = 0.0
    predictions = model.predict(x_valid, batch_size=1)
    txt_name = 'models_val_outputs/' + '.'.join(str(model_list[i]).strip().split('.')[0:-1]) + '.txt'
    to_txt = np.savetxt(txt_name, predictions)

    for i, prediction in tqdm(enumerate(predictions)):
        pred = np.argmax(prediction)
        true = np.argmax(y_valid[i])
        num_correct += (pred == true)
    print("val_acc: ", num_correct / n_valid)


#  测试集test输出阶段
for i, model in enumerate(model_list):
    if model[0] == 'd':
        model = Model(inputs=base_densenet.input, outputs=predictd)
    elif model[0] == 'r':
        model = Model(inputs=base_resnet.input, outputs=predictr)
    model.load_weights(model_dir + str(model_list[i]))

    x_test = load_test_data('dataset/test_imglist.txt', width)

    n_test = len(x_test)
    print('num of valid:', n_test)

    num_correct = 0.0
    predictions = model.predict(x_test, batch_size=1)
    txt_name = 'models_test_outputs/' + '.'.join(str(model_list[i]).strip().split('.')[0:-1]) + '.txt'
    to_txt = np.savetxt(txt_name, predictions)










