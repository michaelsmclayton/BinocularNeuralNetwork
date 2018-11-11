import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from initialiseGaborFilters import initialiseGaborFilters
from loadData import loadAndPreprocessData

'''             THINGS TO DO
 - Find a way of definitely saving weights and biases
 - Change the cost function so that it matches that of Goncalves
 - From saved weights, look at the filter kernals
 - Read relevant parts of paper again to see if you can replicate
'''

# Constant features of the input data
numberOfEyes = 2
inputImageSize = 30

# Model parameters
learningRate = 0.01
numberOfOutputs = 2
batchSize = 100
# nEpochs = 200

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
W1 = initialiseGaborFilters(numberOfFilters, numberOfEyes, filterSize)
b1 = tf.Variable(np.zeros((numberOfFilters,)), trainable=True, name='b1', dtype=tf.float32) # initialse biases with zeros (with one bias for each filter)
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
    conv_with_b = tf.nn.bias_add(conv, b) # Add the bias term (noting that there are still 28 filters)
    conv_out = tf.nn.relu(conv_with_b) # Pass through a rectified non-linear function
    maxPool = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID') # Perform max-pooling
    return maxPool # Return output

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

    ###########################################################
    #                            Setup
    ###########################################################
    summary_writer = tf.summary.FileWriter('summaries/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    # Function to get data mini-batch
    def getBatchData(data, batchIndex):
        return data[batchIndex * batchSize: (batchIndex + 1) * batchSize]

    # Function to get batch labels
    def getBatchLabels(data, batchIndex):
        labels = data[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        currentGroundTruth = tf.one_hot(labels, numberOfOutputs, on_value=1., off_value=0., axis=-1)
        return sess.run(currentGroundTruth)

    # Function to analyse all mini-batches for a given dataset
    def trainAllBatchesOfDataset(xData, yData, numberOfBatches, boredom):

        # Initialise vector to store latest costs and accuracy
        currentCosts = np.zeros(numberOfBatches)
        currentAccuracies = np.zeros(numberOfBatches)

        # Loop over all batches
        for minibatchIndex in xrange(numberOfBatches):
            
            # Get current boredom
            boredom = (epoch - 1) * n_train_batches + minibatchIndex 
            
            # Get data
            currentBatch = getBatchData(data=xData, batchIndex=minibatchIndex) # Get current batch data
            currentGroundTruth = getBatchLabels(data=yData, batchIndex=minibatchIndex) # Get ground truth labels
            
            # Run training op and get accuracy
            _, currentAccuracies[minibatchIndex], currentCosts[minibatchIndex] = sess.run(
                [train_op, accuracy, cost],
                feed_dict={
                    inputImage: currentBatch,
                    groundTruth: currentGroundTruth
                }
            )

            # Break if patience threshold is passed
            if patience <= boredom:
                doneLooping = True
                break

        # Return mean cost/accuracy, and current boredom
        meanCost = np.mean(currentCosts)
        meanAccuracy = np.mean(currentAccuracies)
        return [meanCost, meanAccuracy, boredom]


    ###########################################################
    #                            TRAINING
    ###########################################################
    epoch = 1
    boredom = 0 # Variable to track boredom
    doneLooping = False
    bestValidationCost = np.inf # Initialise to track best validation result
    saver = tf.train.Saver()

    # Early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patienceIncrease = 2  # wait this much longer when a new best is found
    improvementThreshold = 0.995  # threshold for significant improvement
    while (not doneLooping):

        ######################################################
        # Perform one epoch with TRAINING data set
        print 'Training:', 'Epoch', epoch
        [trainingCost, trainingAccuracy, boredom] = trainAllBatchesOfDataset(
            xData=train_set_x,
            yData=train_set_y,
            numberOfBatches=n_train_batches,
            boredom=boredom)
        print 'cost =',  trainingCost, ' accuracy = ', trainingAccuracy


        ######################################################
        # Perform one epoch with VALIDATION data set
        print 'Validation: ', 'Epoch', epoch
        [validationCost, validationAccuracy, boredom] = trainAllBatchesOfDataset(
            xData=valid_set_x,
            yData=valid_set_y,
            numberOfBatches=n_valid_batches,
            boredom=boredom)
        print 'cost = ',  validationCost, 'accuracy = ', validationAccuracy
        print 'boredom = ',  boredom, 'threshold = ', patience
        print ' '

        ######################################################
        # If the validation cost beats the current best
        if validationCost < bestValidationCost:

            # Increase patience if loss improvement is good enough
            if validationCost < bestValidationCost * improvementThreshold:
                patience = max(patience, boredom * patienceIncrease)

            # Save the best validation score and iteration number
            bestValidationCost = validationCost
            saver.save(sess, './modelData/model.ckpt')

        epoch += 1 # Increment epoch




