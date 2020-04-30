import tensorflow as tf
import manifolds


class PManifold(tf.keras.layers.Layer):
    '''
        Model definition for the Persistent Manifold Layer
        The input to this layer is a peristence diagram with
        its points embedded in a m-dim Euclidean space
    '''

    def __init__(self, input_shape, output_shape, manifold='poincare'):
        '''
            Initializes layer params, i.e theta's
        '''
        super(PManifold, self).__init__()

        self.num_of_hom = input_shape[0]
        self.max_num_of_points = input_shape[1]

        self.K = output_shape[1]
        self.man_dim = output_shape[2]

        if manifold == 'poincare':
            self.manifold = manifolds.Poincare(man_dim=self.man_dim)
        if manifold == 'euclidean':
            self.manifold = manifolds.Euclidean()
        if manifold == 'lorenz':
            self.manifold = manifolds.Lorenz(man_dim=self.man_dim)

        self.x_o = tf.zeros(shape=(self.man_dim,))  # the fixed point on the manifold
        #self.x_o = self.manifold.project_to_manifold(self.x_o)

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
        tilled_dgm = tf.pad(tilled_dgm, paddings=[[0, 0], [0, 0], [0, self.man_dim - 2]])

        # Replicate lernable vars self.max_num_of_points times
        tilled_theta = tf.tile(self.theta[ind, :, :], multiples=[1, self.max_num_of_points])
        tilled_theta = tf.reshape(tilled_theta, shape=[-1, self.man_dim])

        # Transform to manifold
        x = self.manifold.parametrization(tilled_dgm)

        # Add lernable vars
        x = tf.add(x, tilled_theta)

        # Transfer to tangent space
        tangent_x = self.manifold.log_map_x(self.x_o, x)

        # Reshaping not really needed, TF might complain for unknown shape that's why we do it
        reshaped_tangent_x = tf.reshape(tangent_x,
                                        shape=[-1, self.max_num_of_points,
                                               self.K, self.man_dim])
        # Sum out diagram points
        sums = tf.reduce_sum(reshaped_tangent_x, axis=1)

        # Transform back to manifold
        x_dgm = self.manifold.exp_map_x(self.x_o, sums)

        # Transform to eucledian
        y_dgm = self.manifold.chart(x_dgm)

        return tf.reshape(y_dgm, shape=[-1, self.K, self.man_dim])

    def get_config(self):
        '''
            Set's the vars of the class. Overrides the Keras method layer and used in case we want
            to save the model. Doesn't really work cuz the manifold class is not serializable.
            Put it here to avoid Keras errors
        '''
        config = super().get_config().copy()
        config.update({
            'projection_bases': self.K,
            'num_of_hom': self.num_of_hom,
            'max_num_of_points': self.max_num_of_points,
            'man_dim': self.man_dim,
            'manifold': self.manifold,
            'x_0': self.x_o,
            'theta': self.theta
        })
        return config

    def call(self, inputs):
        '''
            Call method of Keras Layers
        '''
        # Get the diagrams for the two homology classes
        # Two classes are sufficient for images/graphs
        # TODO generalize to m classes in the future

        dgm_0 = inputs[:, 0, :, :]  # zero-th homology class
        dgm_1 = inputs[:, 1, :, :]  # first homology class

        # Get and concat outputs
        out_0 = self.process_dgm(dgm_0, 0)
        out_1 = self.process_dgm(dgm_1, 1)
        out_0 = tf.expand_dims(out_0, axis=1)
        out_1 = tf.expand_dims(out_1, axis=1)

        return tf.concat([out_0, out_1], axis=1)
