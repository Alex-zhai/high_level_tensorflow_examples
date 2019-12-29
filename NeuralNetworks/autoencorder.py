# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/20 15:09

from __future__ import print_function, absolute_import, division
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import mnist
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# we choose first 20 images to do reconstruction operation
x_re = x_test[:20]


def parse_map_fn(feature):
    feature = tf.cast(tf.reshape(feature["img_raw"], [28 * 28]), tf.float32)
    return feature


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_train}))
    dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x = iterator.get_next()
    return {"img_raw": batch_x}


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_test}))
    dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x = iterator.get_next()
    return {"img_raw": batch_x}


def re_input_fn(batch_size=20):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_re}))
    dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x = iterator.get_next()
    return {"img_raw": batch_x}


def encoder(x):
    x = layers.Dense(256, activation=tf.nn.sigmoid)(x)
    e_out = layers.Dense(128, activation=tf.nn.sigmoid)(x)
    return e_out


def decoder(x):
    x = layers.Dense(256, activation=tf.nn.sigmoid)(x)
    d_out = layers.Dense(28 * 28, activation=tf.nn.sigmoid)(x)
    return d_out


def model_fn(features, mode, params):
    features = features["img_raw"]
    e_out = encoder(features)
    d_out = decoder(e_out)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "pred_imgs": tf.reshape(d_out, shape=[-1, 28, 28]),
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.mean_squared_error(labels=features, predictions=d_out)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels=features, predictions=d_out)
        eval_metric_spec = {
            'mse': tf.metrics.mean_squared_error(labels=features, predictions=d_out),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    ae_model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=30000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=ae_model, train_spec=train_spec, eval_spec=eval_spec)


def get_reconstructed_images(save_model_path):
    ae_model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    predictions = ae_model.predict(input_fn=lambda: re_input_fn())
    pred_imgs = [p["pred_imgs"] for p in predictions]
    for i, pred_img in enumerate(pred_imgs):
        print(pred_img)
        img = Image.fromarray(pred_img, "L")
        img.save("pred_imgs/" + str(i) + ".png")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    save_model_path = "auto_encoder_model"
    train_and_eval(save_model_path)
    get_reconstructed_images(save_model_path)


if __name__ == '__main__':
    tf.app.run()

