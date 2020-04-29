import os
from PIL import Image

import tensorflow as tf
import numpy as np
import networkx as nx
import urllib.request
from zipfile import ZipFile
import shutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_mnist_data(binirize = False, fashion = False):
    '''
        Uses the keras backend to download and binirize the MNIST images
    :return: train_images, train_labels, test_images, test_labels
    '''
    if fashion:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    else:
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


def get_mpeg_data(dir='datasets/mpeg7/', new_size=64, train_split = 0.9):
    '''
        Reads a pre-processes the MPEG7 dataset
    '''
    #abs_dir = os.path.join(os.getcwd(), dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        url = 'http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip'
        filename = 'MPEG7dataset.zip'
        urllib.request.urlretrieve(url, dir + filename)
        with ZipFile(dir + filename, 'r') as zip_obj:
            zip_obj.extractall(dir)
            for image in os.listdir('datasets/mpeg7/original'):
                shutil.move('datasets/mpeg7/original/' + image, dir)
            # Delete non-needed files
            shutil.rmtree('datasets/mpeg7/original')
            os.unlink(dir + filename)
            os.unlink(dir + 'shapedata.fig')
            os.unlink(dir + 'shapedata.eps')
            os.unlink(dir + 'shapedata.gif')
            os.unlink(dir + 'confusions.eps')
            os.unlink(dir + 'confusions.fig')
            os.unlink(dir + 'confusions.gif')

    N = len(os.listdir(dir))
    data = np.zeros(shape=(N, new_size, new_size))
    labels = np.zeros(shape=(N,))
    labels_map = dict()

    for ind, filename in enumerate(os.listdir(dir)):
        im = Image.open(os.path.join(dir, filename))
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

    return x_train, y_train, x_test, y_test

def get_cifar():
    '''
        Downloads the cifar10 dataset
    '''
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.cifar10.load_data()

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

    return x_train, y_train, x_test, y_test

def get_graphs(directory, test_size=0.2):
    '''
        Reads the graphs and labels in the given directory
    '''
    directory = os.path.join('datasets', directory)
    graphs = []
    for file in os.listdir(directory):
        if '.gml' in file:
            graph = nx.readwrite.gml.read_gml(os.path.join(directory, file), label=None)
            graph = nx.relabel.convert_node_labels_to_integers(graph)
            graphs.append(graph)
        elif 'Labels.txt' in file:
            with open(os.path.join(directory, file)) as f:
                labels = f.readlines()
                labels = [label.strip() for label in labels]
    labels = LabelEncoder().fit_transform(labels)
    train_graphs, test_graphs, train_labels, test_labels = \
        train_test_split(graphs, labels, test_size=test_size)
    return train_graphs, train_labels, test_graphs, test_labels





