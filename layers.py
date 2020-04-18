import tensorflow as tf
import utils
from datetime import datetime
import numpy as np


class PManifoldLayer(tf.keras.layers.Layer):
    '''
        Model definition for the Persistent Manifold Layer
        The input to this layer is a peristence diagram with
        its points embedded in a m-dim Euclidean space
    '''

    def __init__(self, K, man_dim, num_of_hom, max_num_of_points):
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
        self.max_num_of_points = max_num_of_points
        self.x_o = tf.zeros(shape=(self.man_dim,))  # the fixed point on the manifold
        self.Poincare = utils.Poincare()

    def build(self, input_shape):
        # Lernable vars
        theta_init = tf.random_uniform_initializer()
        self.theta = tf.Variable(name='theta',
                                 initial_value=theta_init(shape=(self.num_of_hom,
                                                                 self.K, self.man_dim),
                                                          dtype=tf.float32),
                                 trainable=True)
        super(PManifoldLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        '''
            Returns the shape of the output tensor
        '''
        return [-1, self.num_of_hom*self.K*self.man_dim]

    def __cartesian_to_spherical_coordinates(self, point_cartesian):
        '''
            Transform point to 3d spherical cords
            TODO handle zero norm
        '''
        #tf.print(point_cartesian.shape)
        point = tf.convert_to_tensor(value=point_cartesian)
        x, y, z = tf.split(point, num_or_size_splits=3, axis=-1)
        radius = tf.norm(tensor=point, axis=-1, keepdims=True)
        theta = tf.acos(z / radius)
        phi = tf.atan2(y, x)
        return tf.concat((radius, theta, phi), axis=-1)

    def __spherical_to_cartesian_coordinates(self, point_spherical):
        '''
            Transform point to 3d spherical cords
            TODO handle negative radius
        '''
        r, theta, phi = tf.unstack(point_spherical, axis=-1)
        tmp = r * tf.sin(theta)
        x = tmp * tf.cos(phi)
        y = tmp * tf.sin(phi)
        z = r * tf.cos(theta)
        return tf.stack((x, y, z), axis=-1)

    def __process_dgm(self, dgm, ind):
        '''
            Compute the representation of a diagram
        '''

        tilled_dgm = tf.tile(dgm, [1, self.K, 1])
        tilled_dgm = tf.pad(tilled_dgm, [[0, 0], [0, 0], [0, 1]])

        tilled_theta = tf.tile(self.theta[ind,:,:], multiples=[1, self.max_num_of_points])
        tilled_theta = tf.reshape(tilled_theta, shape=[-1, self.man_dim])
        x = self.__cartesian_to_spherical_coordinates(tilled_dgm)
        x = tf.add(tilled_dgm, tilled_theta)
        tangent_x = self.Poincare.tf_log_map_x(self.x_o, x, 1.)
        reshaped_tangent_x = tf.reshape(tangent_x,
                                        shape=[-1, self.max_num_of_points,
                                               self.K, self.man_dim])
        sums = tf.reduce_sum(reshaped_tangent_x, axis=1)
        x_dgm = self.Poincare.tf_exp_map_x(self.x_o, sums, 1.)
        y_dgm = self.__spherical_to_cartesian_coordinates(x_dgm)

        return tf.reshape(y_dgm, shape=[-1, self.K*self.man_dim])

    def call(self, inputs):
        '''
            Call method of Keras Layers
        '''
        input_shape = tf.shape(inputs)
        #tf.print(inputs.shape)
        dgms = inputs
        dgm_0 = tf.squeeze(dgms[:, 0, :, :])  # first homology class
        dgm_1 = tf.squeeze(dgms[:, 1, :, :])  # second one

        out_0 = self.__process_dgm(dgm_0, 0)
        out_1 = self.__process_dgm(dgm_1, 1)

        out = tf.concat([out_0,out_1], axis=1)

        return out


class PManifoldModel(tf.keras.models.Model):
    '''
        Build a Keras model using the PManifoldLayer as input layer
    '''

    def __init__(self, input_shape, num_of_hom, K, units=None):
        super(PManifoldModel, self).__init__()
        if units is None:
            units = [128, 64]
        self.num_of_fil = input_shape[0]
        self.max_num_of_points = input_shape[1]
        self.man_dim = input_shape[2]
        self.num_of_hom = num_of_hom

        self.in_layer = []
        self.inputs = []
        for _ in range(self.num_of_fil):
            pm_layer = PManifoldLayer(K, self.man_dim, self.num_of_hom, self.max_num_of_points)
            cur_input = tf.keras.Input(shape=(self.num_of_hom, self.max_num_of_points, 2))
            self.inputs.append(cur_input)
            self.in_layer.append(pm_layer(cur_input))
            # cur_input = tf.keras.Input(shape=(self.num_of_hom, self.max_num_of_points, 2))
            # self.in_layer.append(cur_input)

        self.in_layer_2 = tf.concat(self.in_layer, axis=1)
        self.flat = tf.keras.layers.Flatten()(self.in_layer_2)
        self.dense1 = tf.keras.layers.Dense(units[0],
                                            activation='relu')(self.flat)
        self.batch_norm = tf.keras.layers.BatchNormalization()(self.dense1)
        self.dense2 = tf.keras.layers.Dense(units[1],
                                            activation='relu')(self.batch_norm)
        self.dropout = tf.keras.layers.Dropout(0.2)(self.dense2)
        self.out_layer = tf.keras.layers.Dense(units=10)(self.dropout)

        self.model = tf.keras.Model(inputs=[self.inputs], outputs=self.out_layer)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
                           # Loss function to minimize
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           # List of metrics to monitor
                           metrics=['sparse_categorical_accuracy']
                           )

    def train(self, x_train, y_train, x_test, y_test):

        # Define the Keras TensorBoard callback.
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        # history = self.model.fit(x_train, y_train,
        #                          batch_size=32,
        #                          epochs=10, steps_per_epoch=900, callbacks=[tensorboard_callback]
        #                          )
        #
        # return
        # Prepare the training dataset.
        batch_size = 64
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # Instantiate a loss function.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        epochs = 3
        for epoch in range(epochs):
            tf.print('Start of epoch %d' % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                x_batch_train = tf.split(x_batch_train, num_or_size_splits=28, axis=1)
                x_batch_train = [tf.squeeze(_) for _ in x_batch_train]

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self.model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                # print(grads)
                # return
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log every 200 batches.
                if step % 50 == 0:
                    tf.print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    tf.print('Seen so far: %s samples' % ((step + 1) * 64))
