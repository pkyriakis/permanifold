import datetime
import tensorflow as tf
import model, math
from tensorboard.plugins.hparams import api as hp

def scheduler(epoch):
    '''
        Scheduler for training rate, discrease exp every 50 epochs
    '''
    rt = math.ceil((epoch + 1) / 5)
    return 0.001 / rt


def train(x_train, y_train, x_test, y_test, train_params, hparams, strategy):
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
    # Get params
    units = train_params['units']
    epochs = train_params['epochs']
    batch_size = train_params['batch_size']

    # Get hyperparams
    man_dim = hparams['man_dim']
    K = hparams['proj_bases']
    manifold = hparams['manifold']

    # Get input shape
    num_of_filtrations = len(x_train)
    num_of_hom = 2  # = x_train[0].shape[1]
    max_num_of_points = []
    for i in range(num_of_filtrations):
        max_num_of_points.append(x_train[i].shape[2])
    in_shape = [num_of_filtrations, num_of_hom, max_num_of_points]

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Build model
    with strategy.scope():
        per_model = model.build_model(input_shape=in_shape,
                                      man_dim=man_dim, K=K,
                                      units=units, manifold=manifold)
        per_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Set up tensorboard and other callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1, write_grads=True)
    rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    hp_callback = hp.KerasCallback(log_dir, hparams=hparams)
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=3)

    # Train
    per_model.fit(x=x_train,
                  y=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[tensorboard_callback,
                             hp_callback],
                  validation_data=(x_test, y_test), shuffle=True)
