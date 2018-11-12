import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
from functions.loadModel import loadModel

# Restore model
print 'Restoring mode parameters'
initialWeights = np.load('./savedData/initialFilters.npy')
[kernelWeights, kernalBiases, outputWeights, outputBiases] = loadModel(baseDir='./bestModel/')

# Horizontal cross-section
weightIndex = 3
horizontalCrossSection = 9
top, bottom = .055, -.05

ax1 = plt.subplot(2, 2, 1)
ax1.plot(initialWeights[horizontalCrossSection,:,0,weightIndex],  '.', color='black', markersize=10)
ax1.plot(initialWeights[horizontalCrossSection,:,0,weightIndex],  color='black', lineWidth=2)
ax1.set_ylim(bottom, top)
ax1.plot([0,19],[0,0], color='black', lineWidth=1)
ax1.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax1.axis('off')

ax2 = plt.subplot(2, 2, 2)
ax2.plot(initialWeights[horizontalCrossSection,:,1,weightIndex],  '.', color='black', markersize=10)
ax2.plot(initialWeights[horizontalCrossSection,:,1,weightIndex],  color='black', lineWidth=2)
ax2.set_ylim(bottom, top)
ax2.plot([0,19],[0,0], color='black', lineWidth=1)
ax2.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax2.axis('off')

ax3 = plt.subplot(2, 2, 3)
ax3.plot(kernelWeights[horizontalCrossSection,:,0,weightIndex],  '.', color='black', markersize=10)
ax3.plot(kernelWeights[horizontalCrossSection,:,0,weightIndex],  color='black', lineWidth=2)
ax3.set_ylim(bottom, top)
ax3.plot([0,19],[0,0], color='black', lineWidth=1)
ax3.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax3.axis('off')

ax4 = plt.subplot(2, 2, 4)
ax4.plot(kernelWeights[horizontalCrossSection,:,1,weightIndex], '.', color='black', markersize=10)
ax4.plot(kernelWeights[horizontalCrossSection,:,1,weightIndex], color='black', lineWidth=2)
ax4.set_ylim(bottom, top)
ax4.plot([0,19],[0,0], color='black', lineWidth=1)
ax4.plot([12,12],[.02,.04], color='black', lineWidth=1.4)
ax4.plot([horizontalCrossSection,horizontalCrossSection],[bottom,top], '--', color='black', lineWidth=1)
ax4.axis('off')
plt.show()
# plt.savefig('weightPlot.png')

# # Function to display weights
# def showWeights(eye, save):
#     plt.figure()
#     rows, cols = 7, 4 # Define enough rows and columns to show the 28 figures 
#     for i in range(np.shape(kernelWeights)[3]):
#         img = kernelWeights[:, :, eye, i]
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(img, cmap='gray', interpolation='none') # Visualize each filter matrix
#         plt.axis('off')
#     if save:
#         filename = './weightFigures/savedWeightsForEye' + str(eye) + '.png'
#         plt.savefig(filename)
#     else:
#         plt.show()

# # Display weights
# showWeights(eye=0, save=False)
# showWeights(eye=1, save=False)