import gzip
import cPickle
import numpy as np

def loadSterogramData(datasetFile):
    ''' Load an Random Dot Stereogram dataset'''

    ##################################################
    # Load data
    ##################################################
    f = gzip.open(datasetFile, 'rb')
    dataset = cPickle.load(f)
    f.close()
    ''' dataset format: tuple(input, label)
    input is a np.ndarray of 4 dimensions (nExamples, 2, image_dims(1), image_dims(2))
    label is a np.ndarray of 1 dimension (nExamples,1)'''

    ##################################################
    # Apply preprocessing
    ################################################## 
    def preprocessData(dataset):
        data_x, data_y = dataset  # Get image data
        test_set_y = np.squeeze(np.maximum(np.sign(data_y), 0)) # Squeeze labels (?)
        disp_labels = np.unique(data_y) # Get range of data labels
        return [(data_x, test_set_y), disp_labels, data_y] # Return data

    # Get and then return data
    processedDataset = preprocessData(dataset)
    test_x, test_y = processedDataset[0]
    disp_labels = processedDataset[1]
    data_y = processedDataset[2]

    # Reshape the data
    transpositionIndices = [0, 2, 3, 1]
    test_x = np.transpose(test_x, transpositionIndices)

    return [test_x, test_y, disp_labels, data_y]
