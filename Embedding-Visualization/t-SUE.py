# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:03:31 2023

@author: ken3

Introduction:
    t-SNE (t-distributed Stochastic Neighbor Embedding) is a 
    dimensionality reduction technique used for visualization and 
    exploration of high-dimensional data. It is a non-linear technique, 
    unlike PCA which is linear, and is based on the idea of 
    mapping high-dimensional data into a lower-dimensional space 
    while preserving the similarities between the data points.

    In probability theory, t-SNE models the high-dimensional data points as 
    a distribution of probabilities over the lower-dimensional space. 
    The t-SNE algorithm minimizes the difference between the joint probabilities
    of the high-dimensional and the low-dimensional data points, 
    which can be expressed as the Kullback-Leibler divergence (also called relative entropy). 
    The t-SNE algorithm maps the data points such that similar points are 
    close to each other in the lower-dimensional space, 
    while dissimilar points are far apart. This results in a more 
    interpretable visualization of the high-dimensional data, 
    making it easier to identify patterns, clusters and relationships.

Reference
"""

import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load MNIST dataset from Tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape data for t-SNE
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Perform t-SNE
tsne = TSNE(n_components=3)
x_train_tsne = tsne.fit_transform(x_train)
x_test_tsne = tsne.fit_transform(x_test)

# Visualize the result
fig, ax = plt.subplots()
scatter = ax.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c=y_train, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes")
ax.add_artist(legend1)
plt.title("t-SNE visualization of MNIST dataset")
plt.savefig("tsne_mnist.png")
plt.show()