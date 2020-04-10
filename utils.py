
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import math


class Poincare:
    '''
        Defines several helper functions needed by the Poincare model
        such as exp and log maps and the coordinate chart and parameterization
    '''
    PROJ_EPS = 1e-5
    EPS = 1e-15
    MAX_TANH_ARG = 15.0

    @staticmethod
    def tf_dot(x, y):
        return tf.reduce_sum(x * y, keepdims=True)

    @staticmethod
    def tf_norm(x):
        return tf.norm(x, keepdims=True)

    @staticmethod
    def tf_lambda_x(x, c):
        return 2. / (1 - c * Poincare.tf_dot(x,x))

    @staticmethod
    def tf_atanh(x):
        return tf.atanh(tf.minimum(x, 1. - Poincare.EPS)) # Only works for positive real x.

    # Real x, not vector!
    @staticmethod
    def tf_tanh(x):
       return tf.tanh(tf.minimum(tf.maximum(x, -Poincare.MAX_TANH_ARG), Poincare.MAX_TANH_ARG))

    @staticmethod
    def tf_project_hyp_vecs(x, c):
        # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
        return tf.clip_by_norm(t=x, clip_norm=(1. - Poincare.PROJ_EPS) / np.sqrt(c), axes=[0])

    @staticmethod
    def tf_mob_add(u, v, c):
        v = v + Poincare.EPS
        tf_dot_u_v = 2. * c * Poincare.tf_dot(u, v)
        tf_norm_u_sq = c * Poincare.tf_dot(u,u)
        tf_norm_v_sq = c * Poincare.tf_dot(v,v)
        denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
        result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
        return Poincare.tf_project_hyp_vecs(result, c)


    @staticmethod
    def tf_exp_map_x(x, v, c):
        v = v + Poincare.EPS # Perturbe v to avoid dealing with v = 0
        norm_v = Poincare.tf_norm(v)
        second_term = (Poincare.tf_tanh(np.sqrt(c) * Poincare.tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
        return Poincare.tf_mob_add(x, second_term, c)

    @staticmethod
    def tf_log_map_x(x, y, c):
        diff = Poincare.tf_mob_add(-x, y, c) + Poincare.EPS
        norm_diff = Poincare.tf_norm(diff)
        lam = Poincare.tf_lambda_x(x, c)
        return (((2. / np.sqrt(c)) / lam) * Poincare.tf_atanh(np.sqrt(c) * norm_diff) / norm_diff) * diff


    @staticmethod
    def tf_parametrization(y, theta):
        '''
            Transforms the Euclidean point y onto the manifold given params theta
        '''
        m = y.shape[0] # manifold dim
        x = []
        for i in range(m):
            if i == 0:
                x_i = tf.multiply(theta[0], tf.norm(y))
            elif i == m - 1:
                den = tf.norm(y[i - 1:])
                x_i = theta[m-1] + tf.acos(y[i - 1] / den)
                if y[m - 1] < 0:
                    x_i = 2*math.pi - x_i
            else:
                den = tf.norm(y[i - 1:])
                x_i = tf.acos(y[i-1]/den) + theta[i]
            x.append(x_i)
        return tf.convert_to_tensor(x)


    @staticmethod
    def tf_chart(x, theta=None):
        '''
            Transforms the manifold point x to the Euclidean space
        '''
        m = x.shape[0]
        if theta == None:
            theta = [1] + [0]*(m-1)
            theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        y = []
        for i in range(m):
            y_i = tf.multiply(x[0],theta[0])
            for j in range(1, i-2):
                y_i = tf.multiply(y_i, tf.sin(x[j] + theta[j]))
            if i == m-1:
                y_i = tf.multiply(y_i, tf.sin(x[i]) + theta[i])
            else:
                y_i = tf.multiply(y_i, tf.cos(x[i]) + theta[i])
            y.append(y_i)
        return tf.convert_to_tensor(y)



def get_mnist_data(binirize = False):
    '''
        Uses the keras backend to downlaoad and binirize the MNIST images
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


### Poincare model helper functions




