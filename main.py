'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''
import math
import pickle

from persistence_diagram import PDiagram
import utils
import numpy as np

train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

## Set the params of the filtrations to extract
# Height filtration
num_of_vects = 6
angles = np.linspace(0, math.pi/2, num_of_vects)
dirs = [[round(math.cos(theta),2),round(math.sin(theta),2)] for theta in angles]
dirs = np.array(dirs)

# Radial filtration
center = np.array([[10,10], [10,20], [20,10], [20,20]])
radius = np.array([5, 10, 15])

# Erosion filtration
n_iter_er = np.array([1,2,3,50])

# Dilation filtration
n_iter_dil = np.array([1,3,5,10,50])

params = {'cubical' : None,
         'height': dirs,
         'radial': {'center' : center,
                    'radius' : radius
                    },
         'erosion': n_iter_er,
         'dilation': n_iter_dil
         }

PD = PDiagram(train_images, params, images_id= 'mnist_train', man_dim=9)
pds = PD.get_pds()
vpds = PD.get_vectorized_pds()

PD = PDiagram(test_images, params, images_id= 'mnist_test', man_dim=9)
pds = PD.get_pds()
vpds = PD.get_vectorized_pds()

