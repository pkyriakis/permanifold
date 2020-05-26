import glob, os
import tensorflow as tf
import model, math
from tensorboard.plugins.hparams import api as hp


def scheduler(epoch):
    '''
        Scheduler for training rate, halve every 25 epochs
    '''
    rt = math.ceil((epoch + 1) / 25)
    return 0.001 / rt


def train(x_train, y_train, x_test, y_test, train_params, hparams, strategy, data_id):
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

    # Get input shape
    num_of_filtrations = len(x_train)
    num_of_hom = 2  # = x_train[0].shape[1]
    max_num_of_points = []
    for i in range(num_of_filtrations):
        max_num_of_points.append(x_train[i].shape[2])
    input_shape = [num_of_filtrations, num_of_hom, max_num_of_points]

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Build model
    with strategy.scope():
        per_model = model.build_model(input_shape=input_shape, hparams=hparams, units=units)
        per_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    print(per_model.summary())

    # Set up model checkpoint files
    model_folder = "K" + str(hparams["proj_bases"]) + "_m" \
                   + str(hparams["man_dim"]) + "_s" + hparams["manifold"]
    model_folder = data_id + '_' + model_folder
    model_file = 'models/' + model_folder + "/weights.{epoch:02d}.h5"
    model_dir = os.path.dirname(model_file)

    # Try to load weights if already there
    if os.path.exists(model_dir):
        list_of_files = glob.glob(model_dir + "/*.h5")
        if list_of_files:
            latest= max(list_of_files, key=os.path.getctime)
            if latest is not None:
                per_model.load_weights(latest)
                print("Loaded model weights, continuing training.")
    else:
        os.makedirs(model_dir) # Create dirs, for some reason Keras callback won't create them

    # Set up model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='loss', save_weights_only=False,
                                                    verbose=1, save_best_only=True, mode='min')

    # Set up tensorboard and other callbacks
    log_dir = "logs/fit/" + model_folder
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning rate scheduler
    rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Hyperparms callback
    hp_callback = hp.KerasCallback(log_dir, hparams=hparams)

    # Train
    per_model.fit(x=x_train,
                  y=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[tensorboard_callback, hp_callback, rate_callback, checkpoint],
                  validation_data=(x_test, y_test), verbose=1)
