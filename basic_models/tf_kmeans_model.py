# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/6/29 14:45

from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np

num_points = 100
dimensions = 2
points = np.random.uniform(0, 1000, [num_points, dimensions])


def train_input_fn():
    x_train_tensor = tf.convert_to_tensor(points, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x_train_tensor)
    dataset = dataset.shuffle(buffer_size=100).repeat(count=1)
    dataset = dataset.batch(100)
    iterator = dataset.make_one_shot_iterator()
    batch_x = iterator.get_next()
    return batch_x


def kmean_model(num_clusters=5, use_mini_batch=False):
    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters, use_mini_batch)
    return kmeans


def train_and_eval(num_iterations=10):
    kmeans = kmean_model()
    previous_centers = None
    for _ in range(num_iterations):
        kmeans.train(train_input_fn)
        cluster_centers = kmeans.cluster_centers()
        if previous_centers is not None:
            print('delta:', cluster_centers - previous_centers)
        previous_centers = cluster_centers
        print('score:', kmeans.score(train_input_fn))
    print('cluster centers:', cluster_centers)
    print("************test************")
    cluster_indices = list(kmeans.predict_cluster_index(train_input_fn))
    for i, point in enumerate(points):
        cluster_index = cluster_indices[i]
        center = cluster_centers[cluster_index]
        print('point:', point, 'is in cluster', cluster_index, 'centered at', center)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval()


if __name__ == '__main__':
    tf.app.run()
