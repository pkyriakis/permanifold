import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf
# from tensorflow.keras.layers import *
import utils
import numpy as np
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = utils.get_mnist_data(rotate_test=True)
plt.imshow(np.squeeze(test_images[1]))
plt.show()


# train_images = np.reshape(train_images, newshape=[-1,28,28,1])
# test_images = np.reshape(test_images, newshape=[-1,28,28,1])
#
# test_images = tf.image.rot90(test_images)
# plt.imshow(np.squeeze(test_images[1]))
# plt.show()
#
# model = tf.keras.Sequential()
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu' , input_shape=(28,28,1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(10))
#
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# print(model.summary())
# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images,test_labels))