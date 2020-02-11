import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from tqdm import tqdm

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import *


def load_data(txt_path, size=224):
    with open(txt_path) as f:
        lines = f.readlines()

    n = len(lines)
    x = np.zeros((n, size, size, 3), np.float32)
    y = np.zeros((n, 70), np.float)

    for i, line in enumerate(tqdm(lines)):
        img_path, age = line[:-1].split(' ')
        x[i] = (cv2.resize(cv2.imread('data/' + img_path), (size, size))[:, :, ::-1].copy() - 127.5) / 127.5
        for j in range(70):
            if j == int(age):
                y[i][j] = 0.7999
            else:
                y[i][j] = 0.0029
        # print(y[i])
    return x, y


width = 224
batch_s = 32

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(width, width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predict = Dense(70, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predict)
# model.summary()


x_train, y_train = load_data('data/train_imglist.txt', width)
x_valid, y_valid = load_data('data/val_imglist.txt', width)

n_train = len(x_train)
n_valid = len(x_valid)
print('num of train:', n_train, 'num of valid:', n_valid)

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True)

train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=batch_s)

weights_dir = 'data/model/weights/densenet121_age_1022/'
os.makedirs(weights_dir, exist_ok=True)
filepath = weights_dir + 'densenet121_{val_acc:.4f}_{val_loss:.4f}.hdf5'

# fit 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
# 注意： 默认为 1
#
# verbose = 0，在控制台没有任何输出
# verbose = 1 ：显示进度条

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='auto', period=1)

#############################################################
adam = Adam(lr=1e-3)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

h = model.fit_generator(
    train_generator,
    steps_per_epoch=n_train // batch_s + 1,
    epochs=40,
    validation_data=(x_valid, y_valid),
    validation_steps=n_valid // batch_s + 1,
    callbacks=[EarlyStopping(patience=4, monitor='val_acc'), checkpoint],
    verbose=1)

#############################################################
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

weights = [x for x in os.listdir(weights_dir) if x[-4:] == 'hdf5']
weights = sorted(weights, reverse=True)
w = weights[0]
print(w)

model.load_weights(weights_dir + w)

h = model.fit_generator(
    train_generator,
    steps_per_epoch=n_train // batch_s + 1,
    epochs=40,
    validation_data=(x_valid, y_valid),
    validation_steps=n_valid // batch_s + 1,
    callbacks=[EarlyStopping(patience=4, monitor='val_acc'), checkpoint],
    verbose=1)

#############################################################
adam = Adam(lr=5e-5)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

weights = [x for x in os.listdir(weights_dir) if x[-4:] == 'hdf5']
weights = sorted(weights, reverse=True)
w = weights[0]
print(w)

model.load_weights(weights_dir + w)

h = model.fit_generator(
    train_generator,
    steps_per_epoch=n_train // batch_s + 1,
    epochs=40,
    validation_data=(x_valid, y_valid),
    validation_steps=n_valid // batch_s + 1,
    callbacks=[EarlyStopping(patience=4, monitor='val_acc'), checkpoint],
    verbose=1)

#############################################################
adam = Adam(lr=2e-5)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

weights = [x for x in os.listdir(weights_dir) if x[-4:] == 'hdf5']
weights = sorted(weights, reverse=True)
w = weights[0]
print(w)

model.load_weights(weights_dir + w)

h = model.fit_generator(
    train_generator,
    steps_per_epoch=n_train // batch_s + 1,
    epochs=40,
    validation_data=(x_valid, y_valid),
    validation_steps=n_valid // batch_s + 1,
    callbacks=[EarlyStopping(patience=4, monitor='val_acc'), checkpoint],
    verbose=1)
