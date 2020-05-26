'''
    Learning Persistent Hyperbolic Representations - NeurIPS Submission
'''
import math
import os
import utils
import argparse

from persistence_diagram import *
from train import train
from tensorboard.plugins.hparams import api as hp

import matplotlib.pyplot as plt

GRAPHS_FROM_FILE = ['COLLAB', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'IMDB-MULTI']

def get_data_images(images_id, rotate_test):
    '''
        Obtains train/test data for the given image set using the provided filtration parameters
    '''
    # Load data
    if images_id == 'fashion-mnist':
        train_images, train_labels, test_images, test_labels = utils.get_mnist_data(fashion=True,
                                                                                    rotate_test=rotate_test)
    elif images_id == 'cifar10':
        train_images, train_labels, test_images, test_labels = utils.get_cifar()
    elif images_id == 'mpeg7':
        train_images, train_labels, test_images, test_labels = utils.get_mpeg_data()
    elif images_id == 'mnist':
        train_images, train_labels, test_images, test_labels = utils.get_mnist_data(rotate_test=rotate_test)
    elif images_id == 'emnist':
        train_images, train_labels, test_images, test_labels = utils.get_emnist_data()
    elif images_id == 'sign':
        train_images, train_labels = utils.get_csv_data('./datasets/sign/sign_mnist_train.csv')
        test_images, test_labels = utils.get_csv_data('./datasets/sign/sign_mnist_test.csv')
    else:
        raise ValueError("Please give valid image dataset name (mnist, emnist, fashion-mnist, mpeg7, cifar10, sign)")

    ## Set the params of the filtrations
    # Height filtration
    num_of_vects = 30
    angles = np.linspace(0, 2 * math.pi, num_of_vects)
    directions = [[round(math.cos(theta), 3), round(math.sin(theta), 3)] for theta in angles]
    directions = np.array(directions)

    # Erosion filtration
    n_iter_er = np.array([1, 2, 3, 50])

    # Dilation filtration
    n_iter_dil = np.array([1, 3, 5, 10, 50])

    # Set filtration params
    params = {'cubical': True,
              'height': directions
              # 'erosion': n_iter_er,
              # 'dilation': n_iter_dil
              }

    # Concat train/test
    N_train = train_images.shape[0]
    images = np.concatenate([train_images, test_images], axis=0)

    # Get PDs for all
    image_pd = ImagePDiagram(images, filtration_params=params, images_id=images_id)
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
    filtrations = ['degree']

    # Try to load if already computed
    graph_pd = GraphPDiagram([], graphs_id=graphs_id, filtrations=filtrations)
    x_train, y_train, x_test, y_test = graph_pd.load_pds()
    if x_train != None:
        return x_train, y_train, x_test, y_test

    # Get train/test graphs
    if graphs_id in GRAPHS_FROM_FILE:
        train_graphs, y_train, test_graphs, y_test =\
            utils.get_graphs_from_file(graphs_id)
    else:
        train_graphs, y_train, test_graphs, y_test = \
            utils.get_graphs_from_dir(graphs_id)

    # Concat to one
    N_train = len(train_graphs)
    graphs = train_graphs + test_graphs

    # Get Pds
    graph_pd.set_graphs(graphs)
    diagrams = graph_pd.get_pds()

    # Split them
    x_train = []
    x_test = []
    for diagram in diagrams:
        x_train.append(diagram[:N_train])
        x_test.append(diagram[N_train:])

    graph_pd.save_pds(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test


def main(args):
    # Parse arguments
    data_type = args.data_type
    data_id = args.data_id
    man_dims = [int(m) for m in args.man_dim.split(",")]
    proj_bases = [int(k) for k in args.proj_bases.split(",")]
    spaces = [s for s in args.spaces.split(",")]
    base_batch = args.batch_size
    epochs = args.epochs
    rotate_test = args.rotate_test
    gpus = args.gpus
    if gpus != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    import tensorflow as tf

    # Get dataset
    if data_type == 'images':
        x_train, y_train, x_test, y_test = \
            get_data_images(data_id, rotate_test)
    else:
        x_train, y_train, x_test, y_test = get_data_graphs(data_id)

    # Mirrored strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    # Set train params
    no_of_classes = len(list(np.unique(y_train)))
    batch_size = base_batch * strategy.num_replicas_in_sync
    train_params = {'units': [256, 128, no_of_classes],
                    'epochs': epochs,
                    'batch_size': batch_size}

    # Set hyperparams to search over
    MAN_DIM = hp.HParam('man_dim', hp.Discrete(man_dims))
    PROJ_BASES = hp.HParam('proj_bases', hp.Discrete(proj_bases))
    MANIFOLD = hp.HParam('proj_bases', hp.Discrete(spaces))

    # Train for all hyperparams
    session_num = 0
    for man_dim in MAN_DIM.domain.values:
        for proj_bases in PROJ_BASES.domain.values:
            for manifold in MANIFOLD.domain.values:
                hparams = {
                    'man_dim': man_dim,
                    'proj_bases': proj_bases,
                    'manifold': manifold,
                    'rotate_test': rotate_test,
                    'dropout': 0.1
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
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", help="the type data; images or graphs")
    parser.add_argument("data_id", help="unique identifier of the data, must match the folder name in the datasets dir")
    parser.add_argument('-m', "--man_dim",
                        help="the manifold dimension to train over, comma-separated list of ints", default="3,6,9,10")
    parser.add_argument('-K', "--proj_bases",
                        help="the number of projection bases, comma-separated list of ints", default="10")
    parser.add_argument("-s", "--spaces",
                        help="manifold(s); comma-separated list, valid options are poincare, lorenz, euclidean",
                        default="poincare,lorenz,euclidean")
    parser.add_argument('-rt','--rotate_test', help="If set the test images are rotated by 90 degs (only valid for images)",
                        action='store_true')
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("-g", "--gpus", help="gpus to use, set to -1 to use all", default="-1")
    args = parser.parse_args()

    main(args)
