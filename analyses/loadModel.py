import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def loadModel():
    # Restore model weights
    with tf.Session() as sess:    
        saver = tf.train.import_meta_graph('../bestModel/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('../bestModel/'))
        kernelWeights, kernalBiases, outputWeights, outputBiases = sess.run(
            ['kernelWeights:0', 'b1:0', 'W_out:0', 'b_out:0']
        )
        return [kernelWeights, kernalBiases, outputWeights, outputBiases]