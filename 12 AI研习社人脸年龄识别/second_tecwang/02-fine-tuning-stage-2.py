from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from libs.network import *
import time
from keras.preprocessing.image import ImageDataGenerator
import cv2
import pandas as pd
from keras.layers import *
from libs.network import classification_loss_stage1, classification_loss_stage2, cross_loss_tec, square_loss_tec

# custom parameters
image_size = 200
n_classes = 70
hidden_dim = 512

# 从 stage1 进行预训练
vgg_model = keras.models.load_model("./saved_models/model2019-10-28-16-25-55.h5", custom_objects={
    "classification_loss_stage1": classification_loss_stage1,
    "square_loss_tec": square_loss_tec,
    "cross_loss_tec": cross_loss_tec})
last_layer = vgg_model.get_layer('avg_pool').output
x = GlobalAveragePooling2D(name="global_average_pooling2d_stage_2")(last_layer)
x = Dropout(0.3, name="dropout_stage_2_2")(x)
out = Dense(n_classes, activation='softmax', name="dense_classification_stage_2")(x)
custom_vgg_model = Model(vgg_model.input, out)

# # 从 stage2 进行预训练
# vgg_model = keras.models.load_model("./saved_models/model2019-10-31-14-26-52.h5", custom_objects={
#     "classification_loss_stage2": classification_loss_stage2,
#     "square_loss_tec": square_loss_tec,
#     "cross_loss_tec": cross_loss_tec})
# custom_vgg_model = vgg_model

# fine-tuning by GAP
print(custom_vgg_model.summary())

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                   validation_split=0)
train_generator = train_datagen.flow_from_directory(
    '../../data/train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset="training")
validation_generator = train_datagen.flow_from_directory(
    '../../data/train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset="validation")

custom_vgg_model.compile(optimizer=keras.optimizers.Adam(lr=1e-7), loss=classification_loss_stage2,
                         metrics=["accuracy", square_loss_tec, cross_loss_tec])
# custom_vgg_model.compile(optimizer=keras.optimizers.Adam(lr=1e-7), loss=keras.losses.categorical_crossentropy, metrics=["accuracy", square_loss_tec, cross_loss_tec])
model_save_callback = keras.callbacks.ModelCheckpoint("./saved_models/model%s.h5" % time.strftime('%Y-%m-%d-%H-%M-%S'),
                                                      monitor="val_loss", verbose=1, save_best_only=False, mode="auto")
custom_vgg_model.fit_generator(train_generator, steps_per_epoch=5000, epochs=2, validation_data=validation_generator,
                               validation_steps=200, callbacks=[model_save_callback])
# custom_vgg_model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=1, validation_data=(x_test, y_test), shuffle=True, callbacks=[model_save_callback])

# keras.models.save_model(custom_vgg_model, "./saved_models/final_model_" + time.strftime('%Y-%m-%d-%H-%M-%S') + ".h5")
