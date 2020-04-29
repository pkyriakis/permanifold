from layer import PManifold
import tensorflow as tf


def build_model(input_shape, man_dim, K, manifold, units=None):
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

    # Setup an input for each filtration
    in_layer = []
    inputs = []
    for i in range(num_of_fil):
        layer_input_shape = [num_of_hom, max_num_of_points[i], 2]
        cur_input = tf.keras.Input(shape=layer_input_shape)

        # Create Persistent Manifold Layers
        pm_layer = PManifold(max_num_of_points=max_num_of_points[i], man_dim=man_dim,
                             num_of_hom=num_of_hom, K=K, manifold=manifold)
        inputs.append(cur_input)
        in_layer.append(pm_layer(cur_input))

    # Flatten
    in_layer_2 = tf.concat(in_layer, axis=1)

    flat = tf.keras.layers.Flatten()(in_layer_2)

    # First dense
    dense1 = tf.keras.layers.Dense(units[0],
                                        activation='relu')(flat)

    # Batch norm
    batch_norm = tf.keras.layers.BatchNormalization()(dense1)

    # Second dense
    dense2 = tf.keras.layers.Dense(units[1],
                                        activation='relu')(batch_norm)

    # Dropout
    dropout = tf.keras.layers.Dropout(0.2)(dense2)

    # Out
    out_layer = tf.keras.layers.Dense(units=units[2])(dropout)

    return tf.keras.Model(inputs=[inputs], outputs=out_layer)
