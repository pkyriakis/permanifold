import datetime
import tensorflow as tf
import model


def train(x_train, y_train, x_test, y_test, man_dim, K, train_params):
    '''
        Sets up and trains the model

        :param x_train, x_test: a list of np.arrays of size
                                (n_samples, num_of_hom, max_num_of_points, 2).
                                Each list element corresponds to a filtration
                                and acts as an input layer in our model.

        :param x_test, y_test: np.arrays of size (n_samples,)

        Note: num_of_hom is set statically to 2. It is known for graphs/images.
              @PManifoldLayer can only handle two homology classes so far.
    '''
    units = train_params['units']
    epochs = train_params['epochs']
    batch_size = train_params['batch_size']

    num_of_filtrations = len(x_train)
    num_of_hom = 2  # = x_train[0].shape[1]
    max_num_of_points = []
    for i in range(num_of_filtrations):
        max_num_of_points.append(x_train[i].shape[2])

    print('Building model.')
    in_shape = [num_of_filtrations, num_of_hom, max_num_of_points]
    per_model = model.build_model(input_shape=in_shape, man_dim=man_dim, K=K, units=units)

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    per_model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    per_model.fit(x=x_train,
                  y=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[tensorboard_callback],
                  validation_data=(x_test, y_test))
