import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Function to generate the convolutional neural network
def generateCNN(inputImage, W1, b1, W_out, b_out):

    # Function to create a convolution layer, with a bias term and non-linear activation function
    def convPoolingLayer(x, W, b, k):
        conv = tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding="VALID", data_format='NHWC',) # Perform convolution
        conv_with_b = tf.nn.bias_add(conv, b) # Add the bias term (noting that there are still 28 filters)
        conv_out = tf.nn.relu(conv_with_b) # Pass through a rectified non-linear function
        maxPool = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID') # Perform max-pooling
        return maxPool # Return output

    # Produce all layers of the CNN
    convOut = convPoolingLayer(x=inputImage, W=W1, b=b1, k=2) # Convolutional layer
    flattenedConvOut = tf.layers.flatten(convOut) # Flatten convolutional results for input i
    out = tf.add(tf.matmul(flattenedConvOut, W_out), b_out) # Pass output through final layer
    return tf.nn.softmax(out) # Use softmax for classification