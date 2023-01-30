# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:54:33 2023

@author: ken3


Introduction
    In PCA, the data is transformed from the original high-dimensional space 
    into a lower-dimensional space, where the principal components are chosen 
    such that they capture the largest amount of variance in the data. 
    The first principal component is the direction in which the data varies 
    the most, and each subsequent component is orthogonal to the previous one 
    and captures the next highest amount of variance in the data.
    
    Mathematically, PCA can be described as an orthogonal transformation that 
    converts a set of correlated variables into a set of 
    linearly uncorrelated variables, known as principal components. 
    The principal components are ordered such that 
    the first principal component explains the maximum variance in the data, 
    the second principal component explains the second highest variance, 
    and so on.

Reference 
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load the MNIST dataset from TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten the data into a single feature vector
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Apply PCA to reduce the dimension to 3 most important features
pca = PCA(n_components=3)
pca.fit(x_train)
x_train_reduced = pca.transform(x_train)

# Visualize the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train_reduced[:, 0], x_train_reduced[:, 1], x_train_reduced[:, 2], c=y_train, cmap='rainbow')
ax.set_xlabel("1st Principal Component")
ax.set_ylabel("2nd Principal Component")
ax.set_zlabel("3rd Principal Component")
plt.show()

# Save the figure
fig.savefig("PCA_mnist.png")

'''
The result of the PCA analysis is a three-dimensional scatter plot, 
where each data point is colored based on its label (i.e. the digit it represents). 
By visualizing the data in this reduced dimension, we can see how well 
the most important features of the original 784-dimensional data capture 
the structure and class separation of the data.
'''