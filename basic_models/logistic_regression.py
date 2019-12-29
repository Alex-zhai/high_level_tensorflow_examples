# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/19 19:59

from __future__ import print_function, absolute_import, division
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_train}, y_train))
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_test}, y_test))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def create_lr_model(input_x):
    w = tf.get_variable(name="weight", shape=[784, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b = tf.get_variable(name="biase", shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())
    logits = tf.matmul(input_x, w) + b
    return logits


def lr_model_fn(features, labels, mode, params):
    features = tf.cast(features["img_raw"], tf.float32)
    # convert 28 *28 shape to 784
    features = tf.reshape(features, [-1, 28 * 28])
    labels = tf.cast(labels, tf.int32)
    logits = create_lr_model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, axis=1),
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(input=logits, axis=1)),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    model_function = lr_model_fn
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
    train_and_eval("lr_model")


if __name__ == '__main__':
    tf.app.run()
