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

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# Load MNIST digits dataset from Sklearn (different from PCA.py)
digits = load_digits()
X = digits.data

# Perform t-SNE to reduce dimension to 3 features
tsne = TSNE(n_components=3)
X_reduced = tsne.fit_transform(X)

# Visualize the 3D result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=digits.target, cmap=plt.cm.get_cmap("jet", 10))
ax.set_xlabel("First Component")
ax.set_ylabel("Second Component")
ax.set_zlabel("Third Component")
plt.show()

# Save the result
fig.savefig("tsne_mnist.png")