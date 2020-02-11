from keras.engine import  Model
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
from libs.network import cross_loss_tec, square_loss_tec, classification_loss_stage1

#custom parameters
image_size = 200
n_classes = 7
hidden_dim = 512

vgg_model = VGGFace(model="senet50", include_top=False, input_shape=(image_size, image_size, 3))
last_layer = vgg_model.get_layer('avg_pool').output

# fine-tuning by GAP
x = GlobalAveragePooling2D()(last_layer)
x = Dropout(0.5, name="dropout_12")(x)
out = Dense(n_classes, activation='softmax', name="dense_classification")(x)
custom_vgg_model = Model(vgg_model.input, out)
custom_vgg_model.compile(optimizer=keras.optimizers.Adam(lr=1e-6), loss=classification_loss_stage1, metrics=["accuracy", square_loss_tec, cross_loss_tec])
print(custom_vgg_model.summary())

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.35)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',
        subset="training")
validation_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',
        subset="validation")


model_save_callback = keras.callbacks.ModelCheckpoint("./saved_models/model%s.h5" % time.strftime('%Y-%m-%d-%H-%M-%S'), monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
custom_vgg_model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50, validation_data=validation_generator, validation_steps=100, callbacks=[model_save_callback])
# custom_vgg_model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=epochs, callbacks=[model_save_callback])
# custom_vgg_model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=1, validation_data=(x_test, y_test), shuffle=True, callbacks=[model_save_callback])
# keras.models.save_model(custom_vgg_model, "./saved_models/final_model_" + time.strftime('%Y-%m-%d-%H-%M-%S') + ".h5")