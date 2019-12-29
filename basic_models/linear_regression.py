# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/6/29 13:45

from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import shutil

train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])


def train_input_fn(batch_size=5):
    dataset = tf.data.Dataset.from_tensor_slices(({"x": train_X}, train_Y))
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(5)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(({"x": test_X}, test_Y))
    dataset = dataset.batch(8)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_linear_model(input_x):
    w = tf.get_variable(name="weight", shape=[1], dtype=tf.float64, initializer=tf.random_normal_initializer())
    b = tf.get_variable(name="biase", shape=[1], dtype=tf.float64, initializer=tf.zeros_initializer())
    pred = tf.multiply(input_x, w) + b
    return pred


def linear_model_fn(features, labels, mode, params):
    features = features["x"]
    pred_value = create_linear_model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "pred": pred_value,
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.mean_squared_error(labels, pred_value)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels, pred_value)
        eval_metric_spec = {
            'mse': tf.metrics.mean_squared_error(labels, pred_value),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    shutil.rmtree(save_model_path, ignore_errors=True)
    model_function = linear_model_fn
    lr_model = tf.estimator.Estimator(
        model_fn=model_function, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn())
    tf.estimator.train_and_evaluate(estimator=lr_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("linear_model")


if __name__ == '__main__':
    tf.app.run()
