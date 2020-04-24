'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''
import math
import os
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from persistence_diagram import *
from sklearn.model_selection import train_test_split

from train import train


def get_data_images(images_id, directions, center, radius, n_iter_er, n_iter_dil):
    '''
        Obtains train/test data for the given image set using the provided filtration paramers
    '''

    # Load data
    if images_id == 'fashion_mnist':
        train_images, train_labels, test_images, test_labels = utils.get_mnist_data(fashion=True)
    elif images_id == 'cifar10':
        train_images, train_labels, test_images, test_labels = utils.get_cifar()
    elif images_id == 'mpeg7':
        train_images, train_labels, test_images, test_labels = utils.get_mpeg_data()
    else:  # Load mnist by default
        train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

    # Set filtration params
    params = {'cubical': None,
              'height': directions,
              'radial': {'center': center,
                         'radius': radius
                         },
              'erosion': n_iter_er,
              'dilation': n_iter_dil
              }

    # Concat train/test
    N_train = train_images.shape[0]
    images = np.concatenate([train_images, test_images], axis=0)

    # Get PDs for all
    image_pd = ImagePDiagram(images, fil_parms=params, images_id=images_id)
    diagrams = image_pd.get_pds()

    # Split them
    x_train = []
    x_test = []
    for diagram in diagrams:
        x_train.append(diagram[:N_train])
        x_test.append(diagram[N_train:])

    y_train = train_labels
    y_test = test_labels

    return x_train, y_train, x_test, y_test


def get_data_graphs(graphs_id):
    '''
        Obtains train/test data for the given graph dataset
    '''
    graphs, labels = utils.get_graphs(graphs_id)
    graph_pd = GraphPDiagram(graphs)
    diagrams = graph_pd.compute_vr_persistence()
    return train_test_split(diagrams, labels, test_size=0.25)


## Set the params of the filtrations
# Height filtration
num_of_vects = 20
angles = np.linspace(0, math.pi / 2, num_of_vects)
directions = [[round(math.cos(theta), 3), round(math.sin(theta), 3)] for theta in angles]
directions = np.array(directions)

# Radial filtration
center = np.array([[10, 10], [10, 20], [15, 15], [20, 10], [20, 20]])
radius = np.array([5, 8, 10, 12, 15])
center = np.array([])
radius = np.array([])

# Erosion filtration
n_iter_er = np.array([1, 2, 3, 50])
#n_iter_er = np.array([])

# Dilation filtration
n_iter_dil = np.array([1, 3, 5, 10, 50])
#n_iter_dil = np.array([])

images_id = 'mnist'

train_params = {'units': [128, 64, 10],
                'epochs': 100,
                'batch_size': 16
                }
# x_train, y_train, x_test, y_test =\
#     get_data_images(images_id, directions, center, radius, n_iter_er, n_iter_dil)

graphs_id = 'IMDB_BINARY'
x_train, x_test, y_train, y_test = get_data_graphs(graphs_id)

train(x_train, y_train, x_test, y_test, man_dim=7, K=20, train_params=train_params)
