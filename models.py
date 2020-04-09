import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import utils

class PManifoldLayer(object):
    '''
        Model definition for the Persistent Manifold Layer
        The input to this layer is a peristence diagram with
        its points embedded in a m-dim Euclidean space
    '''

    def __init__(self, K, m):
        '''
            Initializes layer params, i.e theta's
            :param K: the number of projection bases
            :param m: the dimension of the manifold
        '''
        self.K = K
        self.m = m
        self.x_o = tf.zeros(shape=(self.m,)) # the fixed point on the manifold

        # Lernable vars
        theta_init = tf.random_normal_initializer()
        self.theta = tf.Variable(initial_value=theta_init(shape=(K, m), dtype=tf.float32),
                                 trainable=True, name='theta')

    def __call__(self, dgm):
        '''
            Calculates output
        '''
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(self.theta)
            sum = tf.Variable(tf.zeros(shape=(self.m,), dtype=tf.float32))
            for k in range(self.K):
                for point in dgm: # need to figure out how to deal with points from different homology classes
                    hom = point[0] # first element is the hom. class
                    point = point[1:]
                    y = tf.convert_to_tensor(point, dtype=tf.float32)
                    x = utils.Poincare.tf_parametrization(y, self.theta[k,:])
                    t_vec_x = utils.Poincare.tf_log_map_x(self.x_o, x, 1)
                    sum.assign_add(t_vec_x)

        out = utils.Poincare.tf_exp_map_x(self.x_o, sum, 1)
        print(tp.gradient(t_vec_x,self.theta))

        return out

