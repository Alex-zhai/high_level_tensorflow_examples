# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/2 20:21

import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

max_word_index = max([max(sequence) for sequence in train_data]) + 1
print(max_word_index)


def vectorize_sequences(sequences, dimesion=max_word_index):
    results = np.zeros((len(sequences), dimesion))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


train_data = vectorize_sequences(train_data)


def train_gen():
    while True:
        random_indice = np.random.randint(0, train_data.shape[0])
        yield train_data[random_indice], train_labels[random_indice]


def test_gen():
    while True:
        random_indice = np.random.randint(0, test_data.shape[0])
        yield test_data[random_indice], test_labels[random_indice]


def train_input_fn():
    dataset = tf.data.Dataset.from_generator(train_gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([10000, ]), tf.TensorShape([])))
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def test_input_fn():
    dataset = tf.data.Dataset.from_generator(test_gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([10000, ]), tf.TensorShape([])))
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation=None))
    return model


def model_fn(features, labels, mode, params):
    model = create_model()
    logits = model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits)

        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        eval_metric_spec = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=tf.argmax(input=logits, axis=1))
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    shutil.rmtree(save_model_path, ignore_errors=True)
    es_model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn())
    tf.estimator.train_and_evaluate(estimator=es_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("movie_classify_model")


if __name__ == '__main__':
    tf.app.run()
