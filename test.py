import tensorflow as tf
import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Testing load library

"""
x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.
y_train: uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.
x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.
y_test: uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data.
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# summarize loaded dataset
print('X Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('X Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

print('Y Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Y Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# plot raw pixel data
f, axarr = plt.subplots(2,1)
axarr[0].imshow(x_train[1], cmap=plt.get_cmap('gray'))
axarr[1].imshow(x_test[1], cmap=plt.get_cmap('gray'))

print(y_train[2])
print(y_test[1])
plt.show()