import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import utils, time

class PManifoldLayer(tf.keras.layers.Layer):
    '''
        Model definition for the Persistent Manifold Layer
        The input to this layer is a peristence diagram with
        its points embedded in a m-dim Euclidean space
    '''

    def __init__(self, K, man_dim, num_of_hom):
        '''
            Initializes layer params, i.e theta's
            :param K: the number of projection bases
            :param m: the dimension of the manifold
            :param num_of_hom: int, the number of homology classes
        '''
        super(PManifoldLayer, self).__init__()
        self.K = K
        self.man_dim = man_dim
        self.num_of_hom = num_of_hom
        self.x_o = tf.zeros(shape=(self.man_dim,)) # the fixed point on the manifold


        # Lernable vars
        theta_init = tf.random_normal_initializer()
        self.theta = tf.Variable(name='theta',
                                 initial_value=theta_init(shape=(self.K, self.man_dim),
                                                          dtype=tf.float32),
                                 trainable=True)
        self.class_w = tf.Variable(name='class_weight',
                                   initial_value=theta_init(shape=(self.num_of_hom, ),
                                                            dtype=tf.float32),
                                   trainable=True)

    def compute_output_shape(self, input_shape):
        '''
            Returns the shape of the output tensor
        '''
        return [input_shape[0], 2*self.K*self.man_dim]


    def call(self, input):
        '''
            Calculates output
        '''
        out = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for b in tf.range(input.shape[0]):
            for k in tf.range(self.K):




class PManifoldModel():
    '''
        Build a Keras model using the PManifoldLayer as input layer
    '''
    def __init__(self, input_shape, num_of_hom, K, units=None):
        super(PManifoldModel, self).__init__()
        if units is None:
            units = [256, 128]
        self.num_of_fil = input_shape[0]
        self.max_num_of_points = input_shape[1]
        self.man_dim = input_shape[2] - 1
        self.num_of_hom = num_of_hom

        self.in_layer = []
        inputs = []
        for _ in range(self.num_of_fil):
            pm_layer = PManifoldLayer(K, self.man_dim, self.num_of_hom)
            cur_input = tf.keras.Input(shape=(self.max_num_of_points, num_of_hom))
            inputs.append(cur_input)
            self.in_layer.append(pm_layer(cur_input))

        self.in_layer = tf.concat(self.in_layer, axis=-1)
        self.dense1 = tf.keras.layers.Dense(units[0],
                                            input_shape=(self.num_of_fil*self.man_dim,),
                                            activation='relu')(self.in_layer)
        self.batch_norm = tf.keras.layers.BatchNormalization()(self.dense1)
        self.dense2 = tf.keras.layers.Dense(units[1],
                                            activation='relu')(self.batch_norm)
        self.dropout = tf.keras.layers.Dropout(0.2)(self.dense2)
        self.out_layer = tf.keras.layers.Dense(units=10)(self.dropout)

        model = tf.keras.Model(inputs=[inputs], outputs=self.out_layer)


    def train(self, x_train, y_train, x_test, y_test):
        pass




























