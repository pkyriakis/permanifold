from layer import PManifold
import tensorflow as tf
def build_model(input_shape, K, units=None):
    '''
        Sets up a keras model using the PManifold is first layer
        Architecture past PManifold is as follows:

            Dense -> BatchNorm -> Dense -> Dropout -> Dense (output)

        K is the number of projection bases
    '''
    if units is None:
        units = [128, 64]

    # Get input shapes
    num_of_fil = input_shape[0]
    num_of_hom = input_shape[1]
    max_num_of_points = input_shape[2]
    man_dim = input_shape[3]

    # Setup an input for each filtration
    in_layer = []
    inputs = []
    layer_input_shape = [num_of_hom, max_num_of_points, man_dim]
    for _ in range(num_of_fil):
        pm_layer = PManifold(layer_input_shape, K)
        cur_input = tf.keras.Input(shape=layer_input_shape)
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
    out_layer = tf.keras.layers.Dense(units=10)(dropout)

    return tf.keras.Model(inputs=[inputs], outputs=out_layer)
