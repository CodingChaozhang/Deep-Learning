import keras
from keras.layers import *

def classification_loss_stage1(y_true, y_pred):
    import keras.backend as K
    import keras.backend.tensorflow_backend as C
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    square_loss = K.mean(K.square(K.argmax(y_pred, axis=1) - K.argmax(y_true, axis=1)), axis=-1)
    square_loss = K.cast(square_loss, "float32")

    print("================================================================")
    print("================================================================")
    cross_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    print(square_loss, cross_loss)


    total_loss = square_loss + cross_loss
    # print("================================================================")
    # print(total_loss)
    # total_loss = K.mean(total_loss, axis=-1)
    print("================================================================")
    print(total_loss)

    return total_loss

def classification_loss_stage2(y_true, y_pred):
    import keras.backend as K
    import keras.backend.tensorflow_backend as C
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    square_loss = K.mean(K.square(K.argmax(y_pred, axis=1) - K.argmax(y_true, axis=1)), axis=-1)
    square_loss = K.cast(square_loss, "float32")

    print("================================================================")
    cross_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    print(square_loss, cross_loss)


    total_loss = 0.01 * square_loss + cross_loss
    # print("================================================================")
    # print(total_loss)
    # total_loss = K.mean(total_loss, axis=-1)
    print("================================================================")
    print(total_loss)

    return total_loss

def square_loss_tec(y_true, y_pred):
    import keras.backend as K
    import keras.backend.tensorflow_backend as C
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    square_loss = K.square(K.argmax(y_pred, axis=1) - K.argmax(y_true, axis=1))
    square_loss = K.cast(square_loss, "float32")

    return square_loss

def cross_loss_tec(y_true, y_pred):
    import keras.backend as K
    import keras.backend.tensorflow_backend as C
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    cross_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False)

    return cross_loss
