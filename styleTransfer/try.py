# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:26:52 2023

@author: ken3
"""

import tensorflow as tf
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from PIL import Image

# Load the content and style images
content_image = Image.open("D:/Earth/coding/styleTransfer/digitmon.jpeg")
style_image = Image.open("D:/Earth/coding/styleTransfer/theStarryNight.jpg")

# Preprocess the images
content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)
style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)

# Create a placeholder for the generated image
generated_image = K.placeholder((1, content_array.shape[1], content_array.shape[2], 3))

# Concatenate the content, style, and generated images
input_tensor = K.concatenate([content_array, style_array, generated_image], axis=0)

# Load the VGG19 model
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Define the content and style loss functions
def content_loss(content, generated):
    return K.sum(K.square(generated - content))

def style_loss(style, generated):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    return K.sum(K.square(S - G)) / (4. * (3 ** 2) * (generated.shape[1] ** 2))

# Define the total variation loss function
def total_variation_loss(x):
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# Define the final loss function
loss = content_loss(content_array, generated_image) + style_loss(style_array, generated_image) + total_variation_loss(generated_image)

# Define the optimization algorithm
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Define the training step
@tf.function
def train_step(generated_image):
    with tf.GradientTape() as tape:
        loss_value = loss(generated_image)
    grads, = tape.gradient(loss_value, [generated_image])
    optimizer.apply_gradients([(grads, generated_image)])
    return loss_value

# Train the model
for i in range(1000):
    train_step(generated_image)

# Save the generated image
final_image = np.clip(generated_image.numpy()[0], 0, 255).astype('uint8')
Image.fromarray(final_image).save("final_image.jpg")

# Helper function to compute the gram matrix of an image

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# Add this code to the end of the script to run the training and generate the final image
# Run the training loop for a specified number of steps

for i in range(1000):
    train_step(generated_image)
    if i % 100 == 0:
        print("Step {}/1000, Loss: {:.4f}".format(i, loss(generated_image)))
        
# Clip the pixel values of the generated image to the valid range and convert it to an 8-bit integer
final_image = np.clip(generated_image.numpy()[0], 0, 255).astype('uint8')

# Save the final image 
final_image = Image.fromarray(final_image)
final_image.save("final_image.jpg")
