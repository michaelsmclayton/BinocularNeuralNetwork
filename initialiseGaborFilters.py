import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

''' This script generates 28 (19x19x2) filters for convolution in a CNN. Each filter
is a Gabor patch with different angle. These filters are then fed into a CNN, creating
28x19x19x2 parameters. However, instead of 19x19 parameters, why not just use the
Gabor and sinusoid features as parameters? If you have theta, frequency, Gabor phi
and standard deviation, wouldn't you then have a 28*4*2 matrix to train? A relevant
part of the paper states that "During training we did not constrain the filters to any
particular morphology, neither did we constrain properties such as spatial frequency
selectivity"
'''

# Potential issues
'''
  - The paper says that the phase angle varied from 0 to 1*pi. However, the script has
the angle varying from 0 to 1.5*pi?
'''

def initialiseGaborFilters(numberOfFilters, numberOfEyes, filterSize):

    # Gaussian and sinusoid paraemetrs
    theta = 0. # alge of gaussian pi/2 radians
    frequency = 1 # sinusoud frequency (0.1 cycles/pixel)
    standardDeviation = 0.3 # 3 pixels width of gaussian

    # Initialise filters
    filters = np.zeros((filterSize, filterSize, numberOfEyes, numberOfFilters), dtype=np.float32)

    # Initialise x- and y-dimensions with
    filterRange = 3*standardDeviation # i.e. what is the size of the x and y dimensions of the Gabor filter?
    xDim, yDim = np.meshgrid(
        np.linspace(-filterRange, filterRange, filterSize), # x-dimension (?)
        np.linspace(-filterRange, filterRange, filterSize) # y-dimension (?)
    )
    xDash = xDim * np.cos(theta) + yDim * np.sin(theta) # x*cos(theta) + y*sin(theta)

    # Create an array of angles (length=numberOfFilters), evenly spaced between 0 pi and 1.5 pi
    minimumPhaseAngle = 0. # pi
    maximumPhaseAngle = 1.5 # pi
    allPhases = np.linspace(minimumPhaseAngle, maximumPhaseAngle, numberOfFilters) * np.pi

    # Loop over allPhases
    i = 0 # Initialise i
    gaussian2D = np.exp(-((xDim**2)+(yDim**2))/(2*standardDeviation**2)) # 2D gaussian e( -(x^2 + y^2) / 2 * )
    sinusoidBase = xDash * frequency * 2*np.pi # 
    for currentAngle in allPhases:
        # Create gabor filter with current angle
        filters[:, :, 0, i] = gaussian2D * np.cos(sinusoidBase + currentAngle)

        # Some kind of normalisation?
        filters[:, :, 0, i] = filters[:, :, 0, i] / np.sum(np.fabs(filters[:, :, 0, i]))

        # Copy current gabor filter to other eye (i.e. making left and right filters the same)
        filters[:, :, 1, i] = filters[:, :, 0, i]
        i += 1


    # Display the weights
    W = tf.Variable(initial_value=filters, dtype=tf.float32, trainable=True, name='kernelWeights')

    # Function to display the filter weights
    def show_weights(W, filename=None):
        plt.figure()
        rows, cols = 7, 4 # Define enough rows and columns to show the 28 figures 
        for i in range(np.shape(W)[3]):
            img = W[:, :, 1, i]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray', interpolation='none') # Visualize each filter matrix
            plt.axis('off')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    # # Display the weights
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     # Displauy the intial weights
    #     W_val = sess.run(W)
    #     show_weights(W_val)

    # Return weights
    return W