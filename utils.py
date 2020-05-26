import os
from PIL import Image

import tensorflow as tf
import numpy as np
import pandas as pd

import networkx as nx
import urllib.request
from zipfile import ZipFile
import shutil
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import tensorflow_datasets as tfds

def get_emnist_data(sub = 'letters'):
    '''
        Loads the emnist letters dataset
    '''
    train_images, train_labels = tfds.as_numpy(tfds.load(
        'emnist/letters',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))

    test_images, test_labels = tfds.as_numpy(tfds.load(
        'emnist/letters',
        split='test',
        batch_size=-1,
        as_supervised=True,
    ))
    return train_images, train_labels, test_images, test_labels


def get_mnist_data(binirize=False, fashion=False, rotate_test=False):
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

    if rotate_test:
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        test_images = tf.image.rot90(test_images)
        test_images = test_images.numpy()
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


def get_csv_data(path):
    '''
        Read images in from cvs file; used in sign dataset
    '''
    dataframe = pd.read_csv(path)
    labels = dataframe['label'].values
    labels = LabelEncoder().fit_transform(labels)
    dataframe.drop('label', axis=1, inplace=True)

    images = dataframe.values
    images = images / 255
    images = np.array([np.reshape(i, (28, 28)) for i in images])

    return images, labels

def augment_images(images, labels, N):
    '''
        Augments the given images by applying random shifting, zoom, rotations and horizontal flipping

        N is the number of images to generate
    '''
    size = images.shape[1]
    images = images.reshape(images.shape[0], size, size, 1).astype('float32')
    images, labels = shuffle(images, labels)

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True
    )
    iter = generator.flow(images, labels, batch_size=1)
    train_images = np.zeros(shape=(N,size,size))
    train_labels = np.zeros(shape=(N,))
    for i in range(N):
        im, lb = iter.next()
        train_images[i] = np.squeeze(im)
        train_labels[i] = lb

    return train_images, train_labels


def get_mpeg_data(dir='datasets/mpeg7/', new_size=28, train_split=0.6):
    '''
        Reads a pre-processes the MPEG7 dataset
    '''
    # abs_dir = os.path.join(os.getcwd(), dir)
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

    n = int(train_split * N)
    inds = np.random.permutation(data.shape[0])
    x_train, x_test = data[inds[:n]], data[inds[n:]]
    y_train, y_test = labels[inds[:n]], labels[inds[n:]]

    return x_train, y_train, x_test, y_test


def get_cifar():
    '''
        Downloads the cifar10 dataset
    '''
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = np.zeros((train_images.shape[0], 32, 32))
    x_test = np.zeros((test_images.shape[0], 32, 32))

    for ind in range(train_images.shape[0]):
        im = Image.fromarray(train_images[ind])
        bw_im = im.convert('L')
        x_train[ind] = np.array(bw_im) / 255.

    for ind in range(test_images.shape[0]):
        im = Image.fromarray(test_images[ind])
        bw_im = im.convert('L')
        x_test[ind] = np.array(bw_im) / 255.

    return x_train, y_train, x_test, y_test


def get_graphs_from_dir(graphs_id, test_size=0.2):
    '''
        Reads the graphs and labels in the given directory
    '''
    directory = os.path.join('datasets', graphs_id)
    graphs = []
    label_indices = []
    for file in os.listdir(directory):
        if '.gml' in file:
            # Use networkx to read graph
            # Not the most effiecient, igraph is way faster
            graph = nx.readwrite.gml.read_gml(os.path.join(directory, file), label=None)
            graph = nx.relabel.convert_node_labels_to_integers(graph)
            graphs.append(graph)

            # Store filename cuz it's the index to the label
            fname = file.replace(".gml", "")
            label_indices.append(int(fname))

        elif 'Labels.txt' in file:
            with open(os.path.join(directory, file)) as f:
                labels = f.readlines()
                labels = [label.strip() for label in labels]

    # Get the correct labels
    true_labels = []
    for ind in label_indices:
        true_labels.append(labels[ind])
    labels = true_labels

    # Encode, split and return
    labels = LabelEncoder().fit_transform(labels)
    train_graphs, test_graphs, train_labels, test_labels = \
        train_test_split(graphs, labels, test_size=test_size)
    return train_graphs, train_labels, test_graphs, test_labels


def get_graphs_from_file(graphs_id, test_size=0.2):
    '''
        Read graphs from a single file, valid for COLLAB, REDDIT5k, REDDIT12K
    '''
    directory = os.path.join('datasets', graphs_id)

    # Need to read indicator first
    for file in os.listdir(directory):
        if 'indicator' in file:
            with open(os.path.join(directory, file)) as f:
                lines = f.readlines()
                indicator = [int(line.strip()) for line in lines]
        elif 'labels.txt' in file:
            with open(os.path.join(directory, file)) as f:
                labels = f.readlines()
                labels = [int(label.replace(".0", "").strip()) for label in labels]

    # Init a list of graphs
    graphs = []
    N = max(indicator)
    for _ in range(N):
        graphs.append(nx.Graph())

    # Read graphs
    for file in os.listdir(directory):
        if '_A.txt' in file:
            with open(os.path.join(directory, file)) as f:
                lines = f.readlines()
                for ind, line in tqdm.tqdm(enumerate(lines)):
                    line = line.replace(" ", "")
                    uv = line.split(",")
                    u, v = int(uv[0]), int(uv[1])
                    assert indicator[u - 1] == indicator[v - 1]
                    graph = graphs[indicator[u - 1] - 1]
                    graph.add_node(u)
                    graph.add_node(v)
                    graph.add_edge(u, v)
                    graphs[indicator[u - 1] - 1] = graph

    # Encode, split and return
    labels = LabelEncoder().fit_transform(labels)
    train_graphs, test_graphs, train_labels, test_labels = \
        train_test_split(graphs, labels, test_size=test_size)
    return train_graphs, train_labels, test_graphs, test_labels
