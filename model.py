from layer import PManifold
import tensorflow as tf


def build_model(input_shape, hparams, units=None):
    '''
        Sets up a keras model using the PManifold is first layer
        Architecture past PManifold is as follows:

            Dense -> BatchNorm -> Dense -> Dropout -> Dense (output)

        K is the number of projection bases
    '''
    if units is None:
        units = [256, 128, 10]

    # Get input shapes
    num_of_fil = input_shape[0]
    num_of_hom = input_shape[1]
    max_num_of_points = input_shape[2]

    # Get hyperparams
    man_dim = hparams['man_dim']
    K = hparams['proj_bases']
    manifold = hparams['manifold']

    # Setup an input for each filtration
    in_layer = []
    inputs = []
    for i in range(num_of_fil):
        # Define in/out shapes
        layer_input_shape = [num_of_hom, max_num_of_points[i], 2]
        layer_output_shape = [num_of_hom, K, man_dim]

        # Create keras input tensor
        cur_input = tf.keras.Input(shape=layer_input_shape)

        # Create Persistent Manifold Layer
        pm_layer = PManifold(input_shape=layer_input_shape,
                             output_shape=layer_output_shape, manifold=manifold)

        # Append to lst
        inputs.append(cur_input)
        in_layer.append(pm_layer(cur_input))

    # Concat
    in_layer_2 = tf.concat(in_layer, axis=1)

    # Flatten
    flat = tf.keras.layers.Flatten()(in_layer_2)

    # Batch norm; really needed cuz the outputs of PManifold layer are essentially sums of points
    # in a m-dim manifold, they have realy high vals
    batch_norm = tf.keras.layers.BatchNormalization()(flat)

    # First dense
    dense1 = tf.keras.layers.Dense(units[0],
                                        activation='relu')(flat)

    # Check if there's a second dense
    if units[1] != 0:
        # Second dense
        dense2 = tf.keras.layers.Dense(units[1],
                                            activation='relu')(dense1)
    else:
        dense2 = dense1

    # Dropout
    dropout = tf.keras.layers.Dropout(0.2)(dense2)

    # Out
    out_layer = tf.keras.layers.Dense(units=units[2])(dropout)

    return tf.keras.Model(inputs=[inputs], outputs=out_layer)
