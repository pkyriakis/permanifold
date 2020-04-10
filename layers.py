
import tensorflow as tf
import numpy as np
import utils

class PManifoldLayer(tf.keras.layers.Layer):
    '''
        Model definition for the Persistent Manifold Layer
        The input to this layer is a peristence diagram with
        its points embedded in a m-dim Euclidean space
    '''

    def __init__(self, K, m, num_of_hom):
        '''
            Initializes layer params, i.e theta's
            :param K: the number of projection bases
            :param m: the dimension of the manifold
            :param num_of_hom: int, the number of homology classes
        '''
        super(PManifoldLayer, self).__init__()
        self.K = K
        self.m = m
        self.num_of_hom = num_of_hom
        self.x_o = tf.zeros(shape=(self.m,)) # the fixed point on the manifold


        # Lernable vars
        theta_init = tf.random_normal_initializer()
        self.theta = tf.Variable(name='theta',
                                 initial_value=theta_init(shape=(K, m),
                                                          dtype=tf.float32),
                                 trainable=True)
        self.class_w = tf.Variable(name='class_weight',
                                   initial_value=theta_init(shape=(self.num_of_hom, ),
                                                            dtype=tf.float32),
                                   trainable=True)
    def call(self, dgm):
        '''
            Calculates output
            :param dgm: (None, m+1) np.array; dgm[:,0] is the homology class;
                        the 0-th dim is not know in advance but it is constant in all diagrams.
        '''
        sum = tf.zeros(shape=(self.m,), dtype=tf.float32)
        out = []
        for k in range(self.K):
            ind = 0
            while ind < dgm.shape[0] and np.count_nonzero(dgm[ind]) != 0:
                point = dgm[ind,:]
                hom = int(point[0]) # first element is the hom. class
                point = point[1:]
                y_pnt = tf.convert_to_tensor(point, dtype=tf.float32)
                ## TODO x_pnt has CONSTRAINTS, need to implement this
                x_pnt = utils.Poincare.tf_parametrization(y_pnt, self.theta[k,:])
                t_vec_x = utils.Poincare.tf_log_map_x(self.x_o, x_pnt, 1)
                sum += tf.scalar_mul(self.class_w[hom], t_vec_x)
                ind += 1
            x_dgm = utils.Poincare.tf_exp_map_x(self.x_o, sum, 1)
            y_dgm = utils.Poincare.tf_chart(x_dgm)
            out.append(y_dgm)
        out = tf.concat(out, axis=0)
        return tf.expand_dims(out, axis=0)


class PManifoldModel(tf.keras.Model):
    '''
        Build a Keras model using the PManifoldLayer as input layer
    '''
    def __init__(self, K, m, num_of_hom, units = [256, 128]):
        super(PManifoldModel, self).__init__()
        self.K = K
        self.m = m
        self.num_of_hom = num_of_hom

        self.in_layer = PManifoldLayer(K, m, num_of_hom)
        self.dense1 = tf.keras.layers.Dense(units[0],
                                            input_shape=(K*m,),
                                            activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(units[1],
                                            activation='relu')
        self.droput = tf.keras.layers.Dropout(0.2)
        self.out_layer = tf.keras.layers.Dense(units=10)
        
    def call(self, dgm):
        '''
            Call function
        '''
        x = self.in_layer(dgm)
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.dense2(x)
        x = self.droput(x)
        x = self.out_layer(x)
        return x



























