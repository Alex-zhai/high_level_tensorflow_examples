# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/11 10:17
from keras import layers, models, utils


def create_cnn_model(input_shape=(90, 90, 3), classes=2):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (5, 5), padding="same")(input_layer)
    x = layers.MaxPooling2D(strides=2)(x)
    x = layers.Conv2D(64, (5, 5), padding="same")(x)
    x = layers.MaxPooling2D(strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(classes)(x)
    return models.Model(inputs=input_layer, outputs=logits)


def create_cnn_model_2(input_shape=(90, 90, 3), classes=2):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (11, 11), padding="same")(input_layer)
    x = layers.MaxPooling2D(strides=2)(x)
    x = layers.Conv2D(64, (5, 5), padding="same")(x)
    x = layers.MaxPooling2D(strides=2)(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.MaxPooling2D(strides=2)(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.MaxPooling2D(strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(classes)(x)
    return models.Model(inputs=input_layer, outputs=logits)


if __name__ == '__main__':
    cnn_model = create_cnn_model_2()
    print(cnn_model.summary())
    # utils.plot_model(cnn_model, "cnn_model.png", show_shapes=True)
