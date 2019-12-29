# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/20 11:15

from __future__ import print_function, absolute_import, division
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def parse_map_fn(feature, label):
    feature = tf.cast(tf.reshape(feature["img_raw"], [28 * 28]), tf.float32)
    label = tf.cast(label, tf.int32)
    return feature, label


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_train}, y_train))
    dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return {"img_raw": batch_x}, batch_y


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_test}, y_test))
    dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return {"img_raw": batch_x}, batch_y


def train_and_eval(save_model_path):
    # Feature columns describe how to use the input.
    feature_columns = [tf.feature_column.numeric_column(key="img_raw", shape=[28 * 28])]
    lr_model = tf.estimator.DNNClassifier(hidden_units=[256, 256], feature_columns=feature_columns,
                                          model_dir=save_model_path, n_classes=10)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=lr_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("mp_model")


if __name__ == '__main__':
    tf.app.run()

