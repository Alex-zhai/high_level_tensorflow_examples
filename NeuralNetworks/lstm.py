# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/20 14:35


from __future__ import print_function, absolute_import, division
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def parse_map_fn(feature, label):
    # 28*28 => 28*28*1
    feature = tf.cast(feature["img_raw"], tf.float32)
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


def create_lstm_model(input_shape, num_classes=10):
    input = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(input)
    x = layers.LSTM(128)(x)
    logits = layers.Dense(num_classes)(x)
    return models.Model(inputs=input, outputs=logits)


def lstm_model_fn(features, labels, mode, params):
    features = features["img_raw"]
    model = create_lstm_model(input_shape=[28, 28])
    logits = model(features)
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
    model_function = lstm_model_fn
    lr_model = tf.estimator.Estimator(
        model_fn=model_function, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(), )
    tf.estimator.train_and_evaluate(estimator=lr_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("lstm_model")


if __name__ == '__main__':
    tf.app.run()
    # model = create_cnn_model([28, 28])
    # print(model.summary())
