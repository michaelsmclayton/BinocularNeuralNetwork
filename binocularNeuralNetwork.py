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
nEpochs = 200

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

# Transpose the matrices into the right shape
transpositionIndices = [0, 2, 3, 1]
train_set_x = np.transpose(train_set_x, transpositionIndices)
valid_set_x = np.transpose(valid_set_x, transpositionIndices)
test_set_x = np.transpose(test_set_x, transpositionIndices)
print('test = ', train_set_x.shape)

# Compute the number of minibatches for training, validation and testing
n_train_batches = train_set_x.shape[0]
n_valid_batches = valid_set_x.shape[0]
n_test_batches = test_set_x.shape[0]
n_train_batches /= batchSize
n_valid_batches /= batchSize
n_test_batches /= batchSize


######################################################################################
#                       DEFINE WEIGHTS AND BIASES FOR EACH LAYER
######################################################################################

#######################################
# Initialise input and output tensors
#######################################
inputImage = tf.placeholder(shape=[batchSize, inputImageSize, inputImageSize, numberOfEyes], dtype=tf.float32, name='input')
groundTruth = tf.placeholder(shape=[batchSize, numberOfOutputs], dtype=tf.float32, name='groundTruth')

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
    return tf.nn.softmax(out) # Use softmax for classification
model_op = model()


######################################################################################
#                               MEASURE PERFORMANCE
######################################################################################

# Define the cost function using softmax_cross_entropy_with_logits_v2()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_op, labels=groundTruth))
tf.summary.scalar('cost', cost)

# Define the training op to minimise the cost function
train_op = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

# Check if groundTruth is correct
correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(groundTruth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

merged = tf.summary.merge_all()


######################################################################################
#                               TRAIN THE CLASSIFIER
######################################################################################
with tf.Session() as sess:
    # Setup
    summary_writer = tf.summary.FileWriter('summaries/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    # Number of batches
    numberOfBatches = n_train_batches

    # Perform analysis loop
    print('batch size', batchSize)
    for epoch in range(0, nEpochs):
        print('EPOCH', epoch)

        i = 0
        for minibatchIndex in xrange(numberOfBatches):

            # Get current batch data
            currentBatch = train_set_x[minibatchIndex * batchSize: (minibatchIndex + 1) * batchSize]
            
            # Get ground truth labels
            labels = train_set_y[minibatchIndex * batchSize: (minibatchIndex + 1) * batchSize]
            onehot_labels = tf.one_hot(labels, numberOfOutputs, on_value=1., off_value=0., axis=-1)
            onehot_vals = sess.run(onehot_labels)
            currentGroundTruth = onehot_vals

            # Run training operation and get accuracy
            _, accuracy_val, summary = sess.run([train_op, accuracy, merged], feed_dict={inputImage: currentBatch, groundTruth: currentGroundTruth})
            summary_writer.add_summary(summary, i)
            i += 1
            print(sess.run(W1[1,1,1,1]))
            #print(minibatchIndex, accuracy_val)

        
        # print('DONE WITH EPOCH')
