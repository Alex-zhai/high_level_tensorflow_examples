# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/6/29 15:59


from __future__ import print_function, division, absolute_import

import tensorflow as tf
import shutil
import numpy as np
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

eval_nums = x_test.shape[0]
features_nums = x_train.shape[1]
feature_names = ["feature" + str(i) for i in range(features_nums)]


def train_input_fn(batch_size=32):
    features_np_list = np.split(x_train, features_nums, axis=1)
    features = {
        feature_name: tf.constant(features_np_list[i])
        for i, feature_name in enumerate(feature_names)
    }
    dataset = tf.data.Dataset.from_tensor_slices((features, y_train))
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def eval_input_fn():
    features_np_list = np.split(x_test, features_nums, axis=1)
    features = {
        feature_name: tf.constant(features_np_list[i])
        for i, feature_name in enumerate(feature_names)
    }
    dataset = tf.data.Dataset.from_tensor_slices((features, y_test))
    dataset = dataset.batch(eval_nums)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_y = iterator.get_next()
    return batch_x, batch_y


def get_bucket_boundaries(x):
    """Returns bucket boundaries for feature by percentiles."""
    return np.unique(np.percentile(x, range(0, 100))).tolist()


def get_feature_columns(features):
    # print(feature_names)
    numeric_feature_columns = [tf.feature_column.numeric_column(key=feature_name) for feature_name in feature_names]
    feature_columns = [
        tf.feature_column.bucketized_column(numeric_feature_columns[i],
                                            boundaries=get_bucket_boundaries(features[:, i]))
        for i in range(features_nums)]
    return feature_columns


def get_tree_estimator(model_dir):
    columns = get_feature_columns(x_train)
    tree_estimator = tf.estimator.BoostedTreesRegressor(feature_columns=columns, n_batches_per_layer=5,
                                                        model_dir=model_dir, n_trees=50, max_depth=6,
                                                        learning_rate=0.2)
    return tree_estimator


def train_and_eval(model_dir):
    shutil.rmtree(model_dir, ignore_errors=True)
    tree_estimator = get_tree_estimator(model_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=tree_estimator, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("boosted_tree_model")


if __name__ == '__main__':
    tf.app.run()
