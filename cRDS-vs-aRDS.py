import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions.loadModel import loadModel
from functions.loadStereogramData import loadSterogramData
from functions.generateCNN import generateCNN
import scipy.stats as st

# Load stereogram data
print 'Loading datasets...'
[cRDS_x, cRDS_y, cRDSLabels, cRDS_disparities] = loadSterogramData('./rawData/crds_mixed_norm.pkl.gz')
[aRDS_x, aRDS_y, aRDSLabels, aRDS_disparities] = loadSterogramData('./rawData/ards_mixed_norm.pkl.gz')
'''Note that _y gives near vs. far, while labels gives specific disparity (-20 - 20)'''

# Restructure labels to one-hot encoding
numberOfEntries = cRDS_x.shape[0]
def oneHot(data):
    oneHotMatrix = np.zeros([numberOfEntries, 2])
    for i in range(numberOfEntries):
        if data[i] == 0:
            oneHotMatrix[i,:] = [1,0]
        elif data[i] == 1:
            oneHotMatrix[i,:] = [0,1]
    return oneHotMatrix
cRDS_y = oneHot(cRDS_y)
aRDS_y = oneHot(aRDS_y)

# Crop the stereograms to match previous input (i.e. 30x30 pixels)
croppedSize = 30
cRDS_x = cRDS_x[:,0:croppedSize, 0:croppedSize, :]
aRDS_x = aRDS_x[:,0:croppedSize, 0:croppedSize, :]

# Restore model from saved
print 'Restoring mode parameters...'
[W1, b1, w_out, b_out] = loadModel(baseDir='./bestModel/')

# Generate CNN using restored parameters
print 'Generating CNN...'
numberOfOutputs = 2
batchSize = cRDS_y.shape[0]
inputImage = tf.placeholder(shape=cRDS_x.shape, dtype=tf.float32, name='input')
groundTruth = tf.placeholder(shape=[batchSize, numberOfOutputs], dtype=tf.float32, name='groundTruth')
model = generateCNN(inputImage, W1, b1, w_out, b_out)

# Functions to check performance on new data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=groundTruth))
tf.summary.scalar('cost', cost)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(groundTruth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Get performance
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    [cRDS_activations, correlatedAccuracy, correlatedCost] = sess.run(
        [model, accuracy, cost], feed_dict={inputImage: cRDS_x, groundTruth: cRDS_y})
    [aRDS_activations, anticorrelatedAccuracy, anticorrelatedCost] = sess.run(
        [model, accuracy, cost], feed_dict={inputImage: aRDS_x, groundTruth: aRDS_y})
print('Correlated RDS accuracy = ', correlatedAccuracy)
print('Anti-correlated RDS accuracy = ', anticorrelatedAccuracy)

# Get activations for each disparity
outputNeuron = 1 # Far unit
disparityLabels = np.unique(cRDS_disparities)
def getActivPerDisp(activations, disparities, outputNeuron) :
    i, disparityActivations = 0, np.zeros([100, disparityLabels.shape[0]])
    for disparity in disparityLabels:
        [indicesOfInterest, _] = np.where(disparities==disparity)
        disparityActivations[:,i] = activations[indicesOfInterest, outputNeuron]
        i += 1
    return disparityActivations
corrDispActiv = getActivPerDisp(cRDS_activations, cRDS_disparities, outputNeuron)
antiDispActiv = getActivPerDisp(aRDS_activations, aRDS_disparities, outputNeuron)

##############################################
# Plot results
###############################################
ax = plt.subplot(111)

# Correlated results
correlatedMean = np.mean(corrDispActiv, axis=0)
correlatedError = st.t.interval(0.95, len(corrDispActiv)-1, loc=np.mean(corrDispActiv, axis=0), scale=st.sem(corrDispActiv, axis=0))
ax.plot(disparityLabels, correlatedMean, marker='', color='green', linewidth=2)
ax.fill_between(disparityLabels, correlatedError[0], correlatedError[1], color='green', alpha=0.5)

# Anti-correlated results
antiMean = np.mean(antiDispActiv, axis=0)
antiError = st.t.interval(0.95, len(antiDispActiv)-1, loc=np.mean(antiDispActiv, axis=0), scale=st.sem(antiDispActiv, axis=0))
ax.plot(disparityLabels, antiMean, marker='', color='purple', linewidth=2)
ax.fill_between(disparityLabels, antiError[0], antiError[1], color='purple', alpha=0.5)

# Show figure
ax.spines['right'].set_visible(False) # Hide the right and top spines
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left') # Only show ticks on the left and bottom spines
ax.xaxis.set_ticks_position('bottom')

# Labels and legend
ax.set_xlabel('Disparity (pixels)', fontsize=12)
ax.set_ylabel('Activity of far unit (a.u.)', fontsize=12)
ax.legend(['cRDS', 'aRDS'], frameon=False, fontsize=12)
plt.savefig("disparityResultsForCorrAndAntiCorr.png")