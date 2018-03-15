'''
 meloalright
'''
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)


# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    # get the xte arr
    arr = ['./A000.png', './A001.png', './A002.png', './A003.png', './A004.png', './A005.png']
    for path in arr:
        image = Image.open('./diy_test_data/{path}'.format(path=path)).convert('L')

        print('\nopening {path}'.format(path=path))

        # pylot show the dmp image
        dmp = np.array(image).reshape(28, 28)
        plt.imshow(dmp, cmap='gray')
        plt.show()

        # get the reshape 1,784 vector
        a = np.array(image).reshape(784, )

        # Get nearest neighbor of a
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: a}) # nn_index - That is the nearest neighbor's index
        nn_label = np.argmax(Ytr[nn_index]) # nn_label - That is the nearest neighbor's label
        print("nn-vector", Ytr[nn_index]) # print the nearest neighbor's vector
        print("nn-label", nn_label) # print the nearest neighbor's label

        content = input("\npress enter to continue")
