import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functions.loadModel import loadModel
from functions.generateCNN import generateCNN

# Define learning rate (much higher here than in original study)
learningRate = 100

# Restore model from saved
print 'Restoring mode parameters...'
W1 = np.load('./savedData/W1.npy')
b1 = np.load('./savedData/b1.npy')
w_out = np.load('./savedData/w_out.npy')
b_out = np.load('./savedData/b_out.npy')

# Create random image as starting input
# randomNumber = np.random.randint(1, high=1000)
np.random.seed(seed=666) # For reproducibility  # 666
randomInput = np.random.rand(1,30,30,2)

# Generate CNN using restored parameters
print 'Generating CNN...'
numberOfOutputs = 2
inputImage = tf.Variable(initial_value=randomInput, dtype=tf.float32, name='input')
model = generateCNN(inputImage, W1, b1, w_out, b_out)

# Define the loss function
loss = -model[:,1]
tf.summary.scalar('loss', loss)

# Define the training op to minimise the loss function
train_op = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss=loss, var_list=(inputImage))

# Train
print('Training...')
numberOfIterations = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(numberOfIterations):
        sess.run(train_op)
        if (i%1000)==0:
            print str(i/1000) + '0%'
    finalInput = sess.run(inputImage)

# Save the results
# np.save('./savedData/optimisedInputImage', finalInput)
# np.save('./savedData/randomInputImage', randomInput)

# Seperate the results
leftEye_old = np.squeeze(randomInput[:,:,:,0])
rightEye_old = np.squeeze(randomInput[:,:,:,1])
leftEye_new = np.squeeze(finalInput[:,:,:,0])
rightEye_new = np.squeeze(finalInput[:,:,:,1])

# Plot results
ax1 = plt.subplot(2, 2, 1)
ax1.imshow(leftEye_old, cmap='gray', interpolation='none') # Visualize each filter matrix
ax1.axis('off')
ax1 = plt.subplot(2, 2, 2)
ax1.imshow(rightEye_old, cmap='gray', interpolation='none') # Visualize each filter matrix
ax1.axis('off')
ax1 = plt.subplot(2, 2, 3)
ax1.imshow(leftEye_new, cmap='gray', interpolation='none') # Visualize each filter matrix
ax1.axis('off')
ax1 = plt.subplot(2, 2, 4)
ax1.imshow(rightEye_new, cmap='gray', interpolation='none') # Visualize each filter matrix
ax1.axis('off')
plt.show()

# # Plot the receptive fields
# finalInput = np.squeeze(np.load('./savedData/optimisedInputImage.npy'))
# randomInput = np.squeeze(np.load('./savedData/randomInputImage.npy'))
randomInput = np.squeeze(randomInput)
finalInput = np.squeeze(finalInput)

# Plot the horizontal cross-section
horizontalCrossSection = 15
top, bottom = 10, -10
ax1 = plt.subplot(2, 2, 1)
ax1.plot(randomInput[horizontalCrossSection,:,0],  '.', color='black', markersize=10)
ax1.plot(randomInput[horizontalCrossSection,:,0],  color='black', lineWidth=2)
ax1.set_ylim(bottom, top)
ax1.plot([0,30],[0,0], color='black', lineWidth=1)
ax1.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax1.axis('off')

ax2 = plt.subplot(2, 2, 2)
ax2.plot(randomInput[horizontalCrossSection,:,1],  '.', color='black', markersize=10)
ax2.plot(randomInput[horizontalCrossSection,:,1],  color='black', lineWidth=2)
ax2.set_ylim(bottom, top)
ax2.plot([0,30],[0,0], color='black', lineWidth=1)
ax2.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax2.axis('off')

ax3 = plt.subplot(2, 2, 3)
ax3.plot(finalInput[horizontalCrossSection,:,0],  '.', color='black', markersize=10)
ax3.plot(finalInput[horizontalCrossSection,:,0],  color='black', lineWidth=2)
ax3.set_ylim(bottom, top)
ax3.plot([0,30],[0,0], color='black', lineWidth=1)
ax3.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax3.axis('off')

ax4 = plt.subplot(2, 2, 4)
ax4.plot(finalInput[horizontalCrossSection,:,1],  '.', color='black', markersize=10)
ax4.plot(finalInput[horizontalCrossSection,:,1],  color='black', lineWidth=2)
ax4.set_ylim(bottom, top)
ax4.plot([0,30],[0,0], color='black', lineWidth=1)
ax4.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax4.axis('off')
plt.show()