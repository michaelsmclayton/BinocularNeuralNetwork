import numpy as np
##################################################################
# Analyse readout weights
##################################################################

outputNeuron = 1 # Far unit
w_out = np.load('readoutWeights.npy')
farUnitWeights = w_out[:,outputNeuron]