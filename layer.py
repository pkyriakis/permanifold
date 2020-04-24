import tensorflow as tf
import utils, manifolds
from datetime import datetime
import numpy as np


class PManifold(tf.keras.layers.Layer):
    '''
        Model definition for the Persistent Manifold Layer
        The input to this layer is a peristence diagram with
        its points embedded in a m-dim Euclidean space
    '''

    def __init__(self, max_num_of_points, man_dim, num_of_hom, K, space = 'poincare'):
        '''
            Initializes layer params, i.e theta's
        '''
        super(PManifold, self).__init__()
        self.K = K
        self.num_of_hom = num_of_hom
        self.max_num_of_points = max_num_of_points
        self.man_dim = man_dim
        self.x_o = tf.zeros(shape=(self.man_dim,))  # the fixed point on the manifold

        if space == 'poincare':
            self.manifold = manifolds.Poincare()

        theta_init = tf.random_uniform_initializer()
        self.theta = tf.Variable(name='theta',
                                 initial_value=theta_init(shape=(self.num_of_hom,
                                                                 self.K, self.man_dim),
                                                          dtype=tf.float32),
                                 trainable=True)

    def process_dgm(self, dgm, ind):
        '''
            Compute the representation of a diagram
        '''

        # Replicate diagram self.K times
        tilled_dgm = tf.tile(dgm, [1, self.K, 1])
        tilled_dgm = tf.pad(tilled_dgm, paddings=[[0,0],[0,0],[0,self.man_dim-2]])

        # Replicate lernable vars self.max_num_of_points times
        tilled_theta = tf.tile(self.theta[ind,:,:], multiples=[1, self.max_num_of_points])
        tilled_theta = tf.reshape(tilled_theta, shape=[-1, self.man_dim])

        # Transform to manifold
        x = self.manifold.tf_parametrization(tilled_dgm, self.man_dim)
        # Add lernable vars
        x = tf.add(x, tilled_theta)

        # Transfer to tangent space
        tangent_x = self.manifold.tf_log_map_x(self.x_o, x, 1.)
        reshaped_tangent_x = tf.reshape(tangent_x,
                                        shape=[-1, self.max_num_of_points,
                                               self.K, self.man_dim])
        # Sum out diagram points
        sums = tf.reduce_sum(reshaped_tangent_x, axis=1)

        # Transform back to manifold
        x_dgm = self.manifold.tf_exp_map_x(self.x_o, sums, 1.)

        # Transform to eucledian
        y_dgm = self.manifold.tf_chart(x_dgm, self.man_dim)
        return tf.reshape(y_dgm, shape=[-1, self.K, self.man_dim])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection_bases': self.K,
            'num_of_hom': self.num_of_hom
        })

    def call(self, inputs):
        '''
            Call method of Keras Layers
        '''
        # Get the diagrams for the two homology classes
        # TODO generalize to m classes
        dgm_0 = inputs[:,0,:,:]  # zero-th homology class
        dgm_1 = inputs[:,1,:,:]  # first homology class

        # Get and concat outputs
        out_0 = self.process_dgm(dgm_0, 0)
        out_1 = self.process_dgm(dgm_1, 1)
        out_0 = tf.expand_dims(out_0, axis=1)
        out_1 = tf.expand_dims(out_1, axis=1)
        out = tf.concat([out_0,out_1], axis=1)

        return out