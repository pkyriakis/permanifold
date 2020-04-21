import os
from PIL import Image

import tensorflow as tf
import numpy as np

def get_mnist_data(binirize = False):
    '''
        Uses the keras backend to downlaad and binirize the MNIST images
    :return: train_images, train_labels, test_images, test_labels
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
    if binirize:
        train_images[train_images >= .5] = 1.
        train_images[train_images < .5] = 0.
        test_images[test_images >= .5] = 1.
        test_images[test_images < .5] = 0.

    return train_images, train_labels, test_images, test_labels


def get_mpeg_data(dir='datasets/mpeg7', new_size=28, train_split = 0.8):
    '''
        Reads a pre-processes the MPEG7 dataset
    '''
    rel_dir = os.path.join(os.getcwd(), dir)
    N = len(os.listdir(rel_dir))
    data = np.zeros(shape=(N, new_size, new_size))
    labels = np.zeros(shape=(N,))
    labels_map = dict()

    for ind, filename in enumerate(os.listdir(rel_dir)):
        im = Image.open(os.path.join(rel_dir, filename))
        new_im = im.resize((new_size, new_size))
        array_img = np.asarray(new_im)
        array_img = array_img / 255.
        data[ind:, :] = array_img

        str_label = filename.split("-")[0]
        if str_label in labels_map:
            labels[ind] = labels_map[str_label]
        else:
            lb = len(labels_map.keys())
            labels[ind] = lb
            labels_map[str_label] = lb

    n = int(train_split*N)
    inds = np.random.permutation(data.shape[0])
    x_train, x_test = data[inds[:n]], data[inds[n:]]
    y_train, y_test = labels[inds[:n]], labels[inds[n:]]

    return x_train, y_train, x_test, y_test, labels_map

def get_cifar():
    '''
        Downloads the cifar10 dataset
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    x_train = np.zeros((train_images.shape[0],32,32))
    x_test = np.zeros((test_images.shape[0], 32, 32))

    for ind in range(train_images.shape[0]):
        im = Image.fromarray(train_images[ind])
        bw_im = im.convert('L')
        x_train[ind] = np.array(bw_im)/255.

    for ind in range(test_images.shape[0]):
        im = Image.fromarray(test_images[ind])
        bw_im = im.convert('L')
        x_test[ind] = np.array(bw_im)/255.

    return x_train, train_labels, x_test, test_labels






