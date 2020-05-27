'''
    Learning Persistent Hyperbolic Representations - NeurIPS Submission
'''
import utils, os
import argparse
import numpy as np
from train import train
from tensorboard.plugins.hparams import api as hp


def main(args):
    # Parse arguments
    data_type = args.data_type
    data_id = args.data_id
    man_dims = [int(m) for m in args.man_dim.split(",")]
    proj_bases = [int(k) for k in args.proj_bases.split(",")]
    spaces = [s for s in args.spaces.split(",")]
    base_batch = args.batch_size
    epochs = args.epochs
    gpus = args.gpus
    if gpus != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    import tensorflow as tf

    # Get dataset
    if data_type == 'images':
        x_train, y_train, x_test, y_test = utils.get_data_images(data_id)
    else:
        x_train, y_train, x_test, y_test = utils.get_data_graphs(data_id)

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
                        help="manifold(s); comma-separated list, valid options are poincare, euclidean",
                        default="poincare,euclidean")
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("-g", "--gpus", help="gpus to use, set to -1 to use all", default="-1")
    args = parser.parse_args()

    main(args)
