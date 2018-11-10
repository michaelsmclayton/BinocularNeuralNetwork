import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from initialiseGaborFilters import initialiseGaborFilters

# Constant features of the input data
numberOfEyes = 2
inputImageSize = 30

# Model parameters
numberOfOutputs = 2
learningRate = 0.001
batchSize = 100
nEpochs = 2

# Filter parameters
numberOfFilters = 28
filterSize = 19


######################################################################################
#                       DEFINE WEIGHTS AND BIASES FOR EACH LAYER
######################################################################################

#######################################
# Initialise input and output tensors
#######################################
inputImage = tf.placeholder(shape=[None, inputImageSize, inputImageSize, numberOfEyes], dtype=tf.float32, name='input')
y = tf.placeholder(shape=[None, numberOfOutputs], name='prediction', dtype=tf.float32)

#######################################
# Layer 0. Convolutional layer (initialised with Gabor filters of differing phases)
#######################################
convolutionFilters = initialiseGaborFilters(numberOfFilters, numberOfEyes, filterSize)
W1 = tf.convert_to_tensor(convolutionFilters, name='W1', dtype=tf.float32)
b1 = tf.convert_to_tensor(np.zeros((numberOfFilters,)), name='b1', dtype=tf.float32) # initialse biases with zeros (with one bias for each filter)

##############################
# Layer 1. Logistic Regression layer
##############################
imgSizeAfterPooling = 6
numberOfFinalWeights = imgSizeAfterPooling * imgSizeAfterPooling * numberOfFilters
W_out = tf.Variable(tf.zeros([numberOfFinalWeights, numberOfOutputs]), name='W_out')
b_out = tf.Variable(tf.zeros([numberOfOutputs]), name='b_out')


######################################################################################
#                   DEFINE THE LAYERS AND NETWORK ARCHITECTURE
######################################################################################

# Function to create a convolution layer, with a bias term and non-linear activation function
def convPoolingLayer(x, W, b, k):
    conv = tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding="VALID", data_format='NHWC',) # Perform convolution
    maxPool = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID') # Perform max-pooling
    conv_with_b = tf.nn.bias_add(maxPool, b) # Add the bias term (noting that there are still 28 filters)
    conv_out = tf.nn.relu(conv_with_b) # Pass through a rectified non-linear function
    return conv_out # Return output

# Function to generate the convolutional neural network
def model():
    convOut = convPoolingLayer(x=inputImage, W=W1, b=b1, k=2) # Convolutional layer
    flattenedConvOut = tf.layers.flatten(convOut) # Flatten convolutional results for input i
    reluOut = tf.nn.relu(flattenedConvOut) # Pass output through rectified non-linear function
    out = tf.add(tf.matmul(reluOut, W_out), b_out) # Pass output through final layer
    return out
model_op = model()
print(model)
