import gzip
import cPickle
import numpy as np

def loadBatchedData(datasetFile, batchSize):

    ##################################################
    # Load data
    ##################################################
    print 'Loading dataset...'
    f = gzip.open(datasetFile, 'rb')
    dataset = cPickle.load(f)
    f.close()

    ##################################################
    print 'Apply preprocessing...'
    ################################################## 
    ''' A function to apply preprocessing (e.g. crop images, normalize
    if needed, adjust labels to the classification problem, smooth images
    (psf of the eye)'''
    def preprocessData(dataset): # dataset format: tuple(input, label)

         # Get image data
        data_x, data_y = dataset

        # Rescale images from -1 to 1 (i.e. instead of 0-255)
        data_x = (data_x/255)*2-1

        # Identify the 'crossed' and 'uncrossed' images (i.e. near vs. far)
        crossed = np.random.permutation(np.argwhere(data_y == 0))
        uncrossed = np.random.permutation(np.argwhere(data_y == 1))

        #############################################################
        # Divide the data into training, validation, and test sets
        #############################################################
        m = 38000  # total number of images (hardcoded)
        m_train = (0.7 * m) / 2 # 70% of the data for training
        m_valid = (0.15 * m) / 2 # 15% for validation
        m_test = (0.15 * m) / 2 # 15% for testing
        '''We divide by 2 to get the number of instances per class'''

        # Convert floats to integers
        m_train = int(m_train)
        m_valid = int(m_valid)
        m_test = int(m_test)

        # Intialise arrays to keep indices of training, validation, and testing images
        trainingIndices = np.zeros((m_train*2,), dtype=int)
        validationIndices = np.zeros((m_valid*2,), dtype=int)
        testingIndices = np.zeros((m_test*2,), dtype=int)

        # Function to get even vs. odd indices from start(i.e. 0) to stop
        def getLabelIndices(stop, start=0, step=2):
            crossedIndices = range(start, stop-1, step)
            uncrossedIndices = range(start+1, stop, step)
            return [crossedIndices, uncrossedIndices]


        # Fill training indices with half crossed and uncrossed images
        [crossedIndices, uncrossedIndices] = getLabelIndices(stop=m_train*2)
        trainingIndices[crossedIndices] = crossed[0:m_train, 0]
        trainingIndices[uncrossedIndices] = uncrossed[0:m_train, 0]

        # Fill validation indices with half crossed and uncrossed images
        [crossedIndices, uncrossedIndices] = getLabelIndices(stop=m_valid*2)
        validationIndices[crossedIndices] = crossed[m_train:m_train+m_valid, 0]
        validationIndices[uncrossedIndices] = uncrossed[m_train:m_train+m_valid, 0]

        # Fill testing indices with half crossed and uncrossed images
        [crossedIndices, uncrossedIndices] = getLabelIndices(stop=m_test*2)
        testingIndices[crossedIndices] = crossed[m_train+m_valid:m_train+m_valid+m_test,0]
        testingIndices[uncrossedIndices] = uncrossed[m_train+m_valid:m_train+m_valid+m_test, 0]

        # Create training sets (data) by filling from training, validation, and testing indices
        train_set_x = data_x[trainingIndices, :, :, :]
        valid_set_x = data_x[validationIndices, :, :, :]
        test_set_x = data_x[testingIndices, :, :, :]

        # Create training sets (labels) by filling from training, validation, and testing indices
        train_set_y = data_y[trainingIndices]
        valid_set_y = data_y[validationIndices]
        test_set_y = data_y[testingIndices]

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

        # Return data
        return [
            train_set_x, train_set_y, n_train_batches,
            valid_set_x, valid_set_y, n_valid_batches,
            test_set_x, test_set_y, n_test_batches
        ]
    
    return preprocessData(dataset)