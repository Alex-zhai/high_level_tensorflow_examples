# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/19 20:46

import tensorflow as tf
from tensorflow.python.keras import layers, models, preprocessing
from tensorflow.python.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# pad data to be the same length
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=256, padding='post', value=0)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=256, padding='post', value=0)


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"raw_text": x_train}, y_train))
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"raw_text": x_test}, y_test))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def create_bi_rnn_model(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Embedding(input_dim=10000, output_dim=128)(input)
    x = layers.Bidirectional(layers.LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(2)(x)
    return models.Model(inputs=input, outputs=logits)


def lr_model_fn(features, labels, mode, params):
    features = features["raw_text"]
    # labels = tf.cast(labels, tf.int32)
    model = create_bi_rnn_model((256, ))
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
    model_function = lr_model_fn
    lr_model = tf.estimator.Estimator(
        model_fn=model_function, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=lr_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("bi_rnn_model")


if __name__ == '__main__':
    tf.app.run()
    # model = create_bi_rnn_model(input_shape=(256,))
    # print(model.summary())