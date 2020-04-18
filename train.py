import datetime
import os, math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import persistence_diagram
import tensorflow as tf
import layers
import numpy as np
import utils

#tf.compat.v1.disable_eager_execution()

man_dim = 2  # Dimension of the manifold
K = 5  # number of projection bases
num_of_filtration = 28

num_of_hom = 2  # number of homology classes; hardcoded for now (we know its 3 for images);
# TODO code function to find it
max_num_of_pnts = 77

batch_size = 64
epochs = 3
save_every = 1


def pad_diagrams(dgms):
    '''
        Adds zeros points to the diagrams to make them of equal size; tensorflow cant handle inputs with varied size
    '''
    # Get the highest number of points in a PD
    max_num_of_pnts = 0
    num_of_dgms = 0
    for ind in dgms.keys():
        num_of_dgms = 0
        for filtration in dgms[ind].keys():
            for par_ind in dgms[ind][filtration].keys():
                dgm = dgms[ind][filtration][par_ind]
                max_num_of_pnts = max(max_num_of_pnts, dgm.shape[0])
                num_of_dgms += 1

    # Pad
    # TODO fix this
    max_num_of_pnts = 77
    N = len(dgms.keys())
    out = np.zeros([N, num_of_dgms, num_of_hom, max_num_of_pnts, man_dim], dtype=np.float32)
    out[:, :, :, 2] = -1
    for ind in dgms.keys():
        cnt = 0
        for filtration in dgms[ind].keys():
            for par_ind in dgms[ind][filtration].keys():
                dgm = dgms[ind][filtration][par_ind]
                for p_ind in range(dgm.shape[0]):
                    hom = int(dgm[p_ind, 2])
                    out[ind, cnt, hom, p_ind, :] = dgm[p_ind, :2]
                cnt += 1
    return out


# Obtain the data
img_id = 'mnist'
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()
train_images = train_images.reshape(train_images.shape[0], 28 * 28).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28 * 28).astype('float32')

## Set the params of the filtrations
# Height filtration
num_of_vects = 6
angles = np.linspace(0, math.pi / 2, num_of_vects)
dirs = [[round(math.cos(theta), 2), round(math.sin(theta), 2)] for theta in angles]
dirs = np.array(dirs)

# Radial filtration
center = np.array([[10, 10], [10, 20], [20, 10], [20, 20]])
radius = np.array([5, 10, 15])

# Erosion filtration
n_iter_er = np.array([1, 2, 3, 50])

# Dilation filtration
n_iter_dil = np.array([1, 3, 5, 10, 50])

params = {'cubical': None,
          'height': dirs,
          'radial': {'center': center,
                     'radius': radius
                     },
          'erosion': n_iter_er,
          'dilation': n_iter_dil
          }
# Get persistence diagrams
pd_train = persistence_diagram.PDiagram(train_images, fil_parms=params, images_id='mnist_train')
pd_test = persistence_diagram.PDiagram(test_images, fil_parms=params, images_id='mnist_test')

# Get train test data
dgms_train = pd_train.get_pds()
dgms_test = pd_test.get_pds()
x_train = pad_diagrams(dgms_train)
x_test = pad_diagrams(dgms_test)
y_train = train_labels
y_test = test_labels
# x_train = np.split(x_train, indices_or_sections= 28, axis=1)
# x_test = np.split(x_test, indices_or_sections= 28, axis=1)
# x_train = [np.squeeze(_) for _ in x_train]
# x_test = [np.squeeze(_) for _ in x_test]

# Set up model
in_shape = [num_of_filtration, max_num_of_pnts, 3]
per_model = layers.PManifoldModel(input_shape=in_shape, num_of_hom=num_of_hom, K=K)

per_model.train(x_train,y_train,x_test,y_test)
