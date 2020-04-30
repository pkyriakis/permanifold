'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''
import math
import os
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import tensorflow as tf

from persistence_diagram import *
from train import train
from tensorboard.plugins.hparams import api as hp


def get_data_images(images_id):
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

    ## Set the params of the filtrations
    # Height filtration
    num_of_vects = 20
    angles = np.linspace(0, 2 * math.pi, num_of_vects)
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

    # Set filtration params
    params = {'cubical': False,
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
    # Get train/test graphs
    train_graphs, train_labels, test_graphs, test_labels = utils.get_graphs(graphs_id)

    # Concat to one
    N_train = len(train_graphs)
    graphs = train_graphs + test_graphs

    # Get PDs for all graphs
    filtrations = ['vr', 'degree', 'avg_path']
    graph_pd = GraphPDiagram(graphs, graphs_id=graphs_id, filtrations=filtrations)
    diagrams = graph_pd.get_pds()

    # Split them
    x_train = []
    x_test = []
    for diagram in diagrams:
        x_train.append(diagram)
        x_test.append(diagram[N_train:])

    y_train = train_labels
    y_test = test_labels

    return x_train, y_train, x_test, y_test


def main(d_type, data_id):
    # Get dataset
    if d_type == 'images':
        x_train, y_train, x_test, y_test = \
            get_data_images(data_id)
    else:
        x_train, y_train, x_test, y_test = get_data_graphs(data_id)

    # Mirrored strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    # Set train params
    base_batch = 64
    batch_size = base_batch * strategy.num_replicas_in_sync
    train_params = {'units': [256, 128, 10],
                    'epochs': 20,
                    'batch_size': batch_size}

    # Set hyperparams to search over
    MAN_DIM = hp.HParam('man_dim', hp.Discrete([7]))
    PROJ_BASES = hp.HParam('proj_bases', hp.Discrete([20]))
    MANIFOLD = hp.HParam('proj_bases', hp.Discrete(['poincare']))

    # Train for all hyperparams
    session_num = 0
    for man_dim in MAN_DIM.domain.values:
        for proj_bases in PROJ_BASES.domain.values:
            for manifold in MANIFOLD.domain.values:
                hparams = {
                    'man_dim': man_dim,
                    'proj_bases': proj_bases,
                    'manifold': manifold
                }
                # Print session info
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h: hparams[h] for h in hparams.keys()})

                # Train
                train(x_train, y_train, x_test, y_test,
                      train_params=train_params, hparams=hparams,
                      strategy=strategy, data_id=data_id)
                session_num += 1


if __name__ == '__main__':
    main('images', 'mnist')
