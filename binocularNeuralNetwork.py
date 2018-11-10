import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from initialiseGaborFilters import initialiseGaborFilters
from loadData import loadAndPreprocessData

# Constant features of the input data
numberOfEyes = 2
inputImageSize = 30

# Model parameters
learningRate = 0.001
numberOfOutputs = 2
batchSize = 100
nEpochs = 2

# Filter parameters
numberOfFilters = 28
filterSize = 19


######################################################################################
#                    IMPORT AND SEGMENT THE BINOCULAR IMAGE DATA
######################################################################################
datasets = loadAndPreprocessData("lytroPatches_30x30.pkl.gz")
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]


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
'''Filtering reduces the image size to (img_height-filter_height+1 , img_width-filter_width+1)'''

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
    out = tf.add(tf.matmul(flattenedConvOut, W_out), b_out) # Pass output through final layer
    return tf.nn.softmax(out)
model_op = model()


# ######################################################################################
# #                               MEASURE PERFORMANCE
# ######################################################################################

# # Define the cost function using softmax_cross_entropy_with_logits_v2()
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_op, labels=y))
# tf.summary.scalar('cost', cost)

# # Define the training op to minimise the cost function
# train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Check if prediction is correct
# correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
