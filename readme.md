# TensorFlow implementation of Binocular Neural Network, adapted from Goncalves & Welchman (2017; Current Biology)
- See [link to paper ('"What Not" Detectors Help the Brain See in Depth')](https://www.cell.com/current-biology/fulltext/S0960-9822(17)30404-9)
- See [link to original code](https://www.repository.cam.ac.uk/handle/1810/263961) (written using Theano)

----
## Initial vs. learned filter kernels
- Equivalent to Figure 2B, in original paper 
![](https://github.com/michaelsmclayton/BinocularNeuralNetwork/raw/master/figures/kernelsBeforeAndAfterTraining.png)

----
## Depth unit activations for correlated vs. uncorrelated RDSs
- Code generating result = 'cRDS-vs-aRDS.py'
- Equivalent to Figure 3B, in original paper
![](https://github.com/michaelsmclayton/BinocularNeuralNetwork/raw/master/figures/disparityResultsForCorrAndAntiCorr.png)

## Optimal images for near-unit activation, learned starting from random noise
- Code generating result = 'findOptimalStimulus.py'
- Equivalent to Figure 4B, in original paper
![](https://github.com/michaelsmclayton/BinocularNeuralNetwork/raw/master/figures/optimisedInputImages.png)
