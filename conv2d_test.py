"""
Try to figure out conv2d

I'm pretty sure my debug output of the official implementation is *not*
actually the output of a conv2d. If I input all ones, it gives different
outputs for different pixels, which isn't possible for conv2d. Same input with
the same filter and biases will give the same output for all pixels (ignoring
padding... so the few next to the edge will differ).
"""
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

if __name__ == "__main__":
    img = tf.ones((1,300,300,3), dtype=tf.float32)
    print(img)
    a = [[1,2,3],[4,5,6],[7,8,9]]
    b = [[ 1.6210438458702975e-16, 1.923642395809956e-16, 1.4181790579561985e-16 ], [ 1.3879451433462423e-16, 1.864754018338877e-16, 1.4711192789040823e-16 ], [ -2.840707299224303e-17, 4.617144500875424e-17, -2.955872678059856e-17 ]]
    filter = tf.convert_to_tensor(np.array(b), dtype=tf.float32)
    filter = tf.expand_dims(tf.expand_dims(filter, axis=0), axis=0)
    strides = (1,2,2,1)
    padding = "SAME"
    conv2d = tf.nn.conv2d(img, filter, strides, padding) + -0.4585958421230316
    print(conv2d)
    print(conv2d.shape)

