# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:29:24 2023

@author: ken3
"""

# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np

# Define the paths for the content and style images
content_path = "D:/Earth/coding/styleTransfer/digitmon.jpeg"
style_path = "D:/Earth/coding/styleTransfer/theStarryNight.jpg"

# Load and preprocess the images
content_image = Image.open(content_path)
content_image = np.array(content_image)
content_image = np.expand_dims(content_image, axis=0)

style_image = Image.open(style_path)
style_image = np.array(style_image)
style_image = np.expand_dims(style_image, axis=0)

# Create the VGG19 model with the pre-trained weights
vgg = VGG16(weights='imagenet', include_top=False)

# Define the layers to be used for the content and style representations
content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Create the feature extractor models for the content and style
content_extractor = Model(inputs=vgg.input, outputs=[vgg.get_layer(layer_name).output for layer_name in content_layers])
style_extractor = Model(inputs=vgg.input, outputs=[vgg.get_layer(layer_name).output for layer_name in style_layers])

# Define the loss functions for the content and style
def content_loss(content, generated):
    return K.sum(K.square(generated - content))

def style_loss(style, generated):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    return K.sum(K.square(S - G)) / (4 * (3**2) * (128**2))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# Define the total loss as the combination of the content and style loss
def total_loss(content_image, style_image, generated_image, alpha=1.0, beta=1.0):
    content_representation = content_extractor(content_image)
    generated_representation = content_extractor(generated_image)
    content_loss_value = content_loss(content_representation, generated_representation)

    style_representation = style_extractor(style_image)
    generated_representation = style_extractor(generated_image)
    style_loss_value = style_loss(style_representation, generated_representation)

    total_loss = alpha * content_loss_value + beta * style_loss_value
    return total_loss

# Define the optimizer and the target image
optimizer = Adam(lr=0.02)
generated_image = K.random_normal(content_image.shape)

# Define the tensorflow variable for the generated image and apply the gradients
generated_image = K.variable(generated_image)
loss = total_loss(K.constant(content_image), K.constant(style_image), generated_image)
updates = optimizer.get_updates(generated_image, [], loss)

# Define the function to train the model
train_step = tf.function([], [], updates)

# Perform the training
num_steps = 100
for step in range(num_steps):
    train_step()

# Convert the final image to a PIL image and save it
final_image = np.squeeze(generated_image.numpy(), axis=0)
final_image = Image.fromarray(np.uint8(final_image))
final_image.save("generated_image.jpg")

# The generated image should now be saved in the specified path.