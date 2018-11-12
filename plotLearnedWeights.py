import numpy as np
import matplotlib.pyplot as plt
from functions.loadModel import loadModel

# Restore model
[kernelWeights, kernalBiases, outputWeights, outputBiases] = loadModel(baseDir='../../bestModel/')
print 'Restoring mode parameters'

# Function to display weights
def showWeights(eye, save):
    plt.figure()
    rows, cols = 7, 4 # Define enough rows and columns to show the 28 figures 
    for i in range(np.shape(kernelWeights)[3]):
        img = kernelWeights[:, :, eye, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', interpolation='none') # Visualize each filter matrix
        plt.axis('off')
    if save:
        filename = './weightFigures/savedWeightsForEye' + str(eye) + '.png'
        plt.savefig(filename)
    else:
        plt.show()

# Display weights
showWeights(eye=0, save=False)
showWeights(eye=1, save=False)