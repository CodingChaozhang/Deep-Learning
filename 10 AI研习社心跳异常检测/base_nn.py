# 导入模块
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from keras import regularizers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model
from sklearn import preprocessing
from keras.optimizers import SGD
from keras import backend as K
from keras.initializers import glorot_uniform,glorot_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from keras.layers import concatenate, Input, Add, Dropout, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

def Convnet(input_shape=(187,1,1),classes=2):

    X_input = Input(input_shape)
    # X = ZeroPadding2D((10,1))(X_input)
    X1 = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
    # X1 = BatchNormalization(axis=3)(X1)
    X1 = Activation('relu')(X1)
    X1 = MaxPooling2D((10,1),strides=(10,1))(X1)
    X2 = Conv2D(64,(21,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
    # X2 = BatchNormalization(axis=3)(X2)
    X2 = Activation('relu')(X2)
    X2 = MaxPooling2D((10,1),strides=(10,1))(X2)
    X3 = Conv2D(64,(31,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
    # X3 = BatchNormalization(axis=3)(X3)
    X3 = Activation('relu')(X3)
    X3 = MaxPooling2D((10,1),strides=(10,1))(X3)
    X = concatenate([X1,X2,X3],axis=3)
    X = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,1),strides=(2,1))(X)
    X = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,1),strides=(2,1))(X)
    X = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,1),strides=(2,1))(X)
    X = Flatten()(X)
    X = Dense(512, kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(512, kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input,outputs=X,name='Convnet')

    return model

def acc_heartbeat(y_true, y_pred):

    acc = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            acc = acc+1

    return acc/len(y_pred)

lis = []
for i in range(188):
    lis.append(str(i))

# train = pd.read_csv("ptbdb_train.csv",header=None,names=lis)
train = pd.read_csv("train.csv",header=None,names=lis)
test = pd.read_csv("ptbdb_test.csv",header=None,names=lis[:-1])

label = train.pop('187').values

label = np_utils.to_categorical(label,2)
train_mean = train.mean(axis=1)
train = (train.values.T/train_mean.values).T
print(train)

# train = train.values

# scaler = preprocessing.StandardScaler().fit(train)

# train = scaler.transform(train)

train = train.reshape((-1,187,1,1))

train_X, test_X, train_y, test_y = train_test_split(train,label,test_size=0.3,random_state=42)
model = Convnet(input_shape=(187,1,1),classes=2)

# sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])


best = 0
patience = 5
no_improved_times = 0
test_mean = test.mean(axis=1)
test = (test.values.T / test_mean.values).T
# test = scaler.transform(test)
test = test.reshape((-1, 187, 1, 1))

for epoch in range(2000):
    model.fit(train_X,train_y,
          batch_size=256,nb_epoch=30,verbose=1,validation_data=(test_X,test_y))

    score = model.evaluate(test_X,test_y,verbose=0)
    y_true = np.argmax(test_y, 1)
    y_pred = model.predict(test_X)
    y_pred = np.argmax(y_pred, 1)
    acc = acc_heartbeat(y_true, y_pred)
    if acc>best:
        best = acc
        test_pred = model.predict(test)
        test_pred = np.argmax(test_pred, 1)
    else:
        no_improved_times = no_improved_times +1
    if no_improved_times>patience:
        break

print(best)
#
# model.fit(train,label,
#       batch_size=128,nb_epoch=8,verbose=1)



test_sub = pd.DataFrame({'id':range(len(test)),'type':test_pred})
test_sub['type'] = test_pred
test_sub['type'] = test_sub['type'].map({0:'0.0',1:'1.0'})
test_sub.to_csv('sub.csv', index=False,header=None)