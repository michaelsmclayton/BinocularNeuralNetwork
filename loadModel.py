import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Restore weights
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('./bestModel/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./bestModel/'))
    kernelWeights, kernalBiases, outputWeights, outputBiases = sess.run(
        ['kernelWeights:0', 'b1:0', 'W_out:0', 'b_out:0']
    )
print(kernelWeights.shape)

# Display weights
def showWeights(eye):
    plt.figure()
    rows, cols = 7, 4 # Define enough rows and columns to show the 28 figures 
    for i in range(np.shape(kernelWeights)[3]):
        img = kernelWeights[:, :, eye, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', interpolation='none') # Visualize each filter matrix
        plt.axis('off')
    filename = './weightFigures/savedWeightsForEye' + str(eye) + '.png'
    plt.savefig(filename)
showWeights(eye=0)
showWeights(eye=1)