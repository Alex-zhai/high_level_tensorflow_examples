# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/20 16:22

from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.datasets import mnist
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
noise_dim = 200  # Noise data points

x_gen = np.random.uniform(-1., 1., size=[10, noise_dim])


def parse_map_fn(feature):
    feature = tf.cast(tf.expand_dims(feature["img_raw"], axis=2), tf.float32)
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
    # Generate noise to feed to the generator
    z = tf.random_uniform(shape=(batch_size, noise_dim), minval=-1, maxval=1)
    # Prepare Targets (Real image: 1, Fake image: 0)
    # The first half of data fed to the generator are real images,
    # the other half are fake images (coming from the generator).
    batch_disc_y = tf.concat([tf.ones(batch_size, tf.int32), tf.zeros(batch_size, tf.int32)], axis=0)
    batch_gen_y = tf.ones(batch_size, tf.int32)
    return {"img_raw": batch_x, "noise_input": z, "disc_y": batch_disc_y, "gen_y": batch_gen_y}


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"img_raw": x_test}))
    dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x = iterator.get_next()
    z = tf.random_uniform(shape=(batch_size, noise_dim), minval=-1, maxval=1)
    batch_disc_y = tf.concat([tf.ones(batch_size, tf.int32), tf.zeros(batch_size, tf.int32)], axis=0)
    batch_gen_y = tf.ones(batch_size, tf.int32)
    return {"img_raw": batch_x, "noise_input": z, "disc_y": batch_disc_y, "gen_y": batch_gen_y}


def re_input_fn(batch_size=10):
    dataset = tf.data.Dataset.from_tensor_slices(x_gen)
    # dataset = dataset.map(parse_map_fn, num_parallel_calls=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x = iterator.get_next()
    return {"noise_input": batch_x}


def generator(x, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.layers.dense(x, units=6 * 6 * 128, activation=tf.nn.tanh)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2, activation=tf.nn.sigmoid)
        return x


def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.tanh)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5, activation=tf.nn.tanh)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 1024, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 2)
        return x


def model_fn(features, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        noise_input = features["noise_input"]
        gen_samples = generator(noise_input)
        predictions = {
            "gen_images": gen_samples,
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    else:
        real_image_input = features["img_raw"]
        noise_input = features["noise_input"]
        disc_y = features["disc_y"]
        gen_y = features["gen_y"]

        # Build Generator Network
        gen_samples = generator(noise_input)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = discriminator(real_image_input)
        disc_fake = discriminator(gen_samples, reuse=True)
        disc_concat = tf.concat([disc_real, disc_fake], axis=0)

        # Build the stacked generator/discriminator, from generator to discriminator
        stacked_gan = discriminator(gen_samples, reuse=True)

        disc_loss = tf.losses.sparse_softmax_cross_entropy(labels=disc_y, logits=disc_concat)
        gen_loss = tf.losses.sparse_softmax_cross_entropy(labels=gen_y, logits=stacked_gan)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            gen_op = optimizer.minimize(gen_loss, var_list=gen_vars, global_step=tf.train.get_or_create_global_step())
            disc_op = optimizer.minimize(disc_loss, var_list=disc_vars,
                                         global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN, loss=disc_loss + gen_loss,
                train_op=tf.group(gen_op, disc_op)
            )

        else:
            loss = disc_loss + gen_loss
            eval_metric_spec = {
                'mean_loss': tf.metrics.mean(loss)
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
            )


def train_and_eval(save_model_path):
    acgan_model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=20000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=acgan_model, train_spec=train_spec, eval_spec=eval_spec)


def get_gen_images(save_model_path, save_images_path):
    acgan_model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    predictions = acgan_model.predict(input_fn=lambda: re_input_fn())
    pred_imgs = [p["gen_images"] for p in predictions]
    for i, pred_img in enumerate(pred_imgs):
        print(pred_img.shape)
        # img = Image.fromarray(pred_img, "L")
        if os.path.exists(save_images_path):
            os.makedirs(save_model_path)
            # img.save(save_images_path + str(i) + ".png")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    save_model_path = "acgan_model"
    train_and_eval(save_model_path)
    save_images_path = "acgan_gen_images/"
    get_gen_images(save_model_path, save_images_path)


if __name__ == '__main__':
    # tf.app.run()
    features = re_input_fn()
    sess = tf.Session()
    print(sess.run(features))
