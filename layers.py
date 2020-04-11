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

    def call(self, input):
        '''
            Calculates output
            :param dgm: (None, None, m+1) np.array; dgm[:,0] is the homology class;
                        first None is the batch size, second is the number of points in the diagram
        '''

        output_batch = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(tf.shape(input)[0]):
            dgm = input[i]
            out = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            sum = tf.zeros(shape=(self.m,), dtype=tf.float32)
            for k in tf.range(self.K):
                j = tf.constant(0)

                while j < tf.shape(dgm)[0] and tf.cast(dgm[j,0], tf.int32) <= 1 and tf.math.count_nonzero(dgm[j,:]) != 0:
                    point = dgm[j,:]
                    hom = tf.cast(point[0], tf.int32) # first element is the hom. class
                    point = point[1:]
                    y_pnt = tf.convert_to_tensor(point, dtype=tf.float32)
                    ## TODO x_pnt has CONSTRAINTS, need to implement this
                    x_pnt = utils.Poincare.tf_parametrization(y_pnt, self.theta[k,:])
                    t_vec_x = utils.Poincare.tf_log_map_x(self.x_o, x_pnt, 1.)
                    sum = tf.add(sum,
                                 tf.scalar_mul(self.class_w[hom], t_vec_x))
                    j = tf.add(j,1)
                x_dgm = utils.Poincare.tf_exp_map_x(self.x_o, sum, 1.)
                y_dgm = utils.Poincare.tf_chart(x_dgm)
                out = out.write(k, y_dgm)
            out = out.stack()
            out = tf.reshape(out, shape=[-1])
            output_batch = output_batch.write(i, out)
            tf.print(i)
        output_batch = output_batch.stack()
        return tf.reshape(output_batch, shape=[-1, self.m*self.K])


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

    @tf.function
    def call(self, dgm, training=False):
        '''
            Call function
        '''
        x = self.in_layer(dgm)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dense2(x)
        x = self.droput(x, training=training)
        x = self.out_layer(x)
        return x



























