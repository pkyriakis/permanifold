import datetime
import os, math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import persistence_diagram
import tensorflow as tf
import layer, model
import numpy as np
import utils

# Model hyperparms
man_dim = 3  # Dimension of the manifold
K = 100  # number of projection bases

## Set the params of the filtrations
# Height filtration
num_of_vects = 14
angles = np.linspace(0, math.pi / 2, num_of_vects)
dirs = [[round(math.cos(theta), 2), round(math.sin(theta), 2)] for theta in angles]
dirs = np.array(dirs)

# Radial filtration
center = np.array([[10, 10], [10, 20], [15,15], [20, 10], [20, 20]])
radius = np.array([5, 8, 10, 12, 15])
center = np.array([])
radius = np.array([])


# Erosion filtration
n_iter_er = np.array([1, 2, 3, 50])
n_iter_er = np.array([])

# Dilation filtration
n_iter_dil = np.array([1, 3, 5, 10, 50])
n_iter_dil = np.array([])

params = {'cubical': None,
          'height': dirs,
          'radial': {'center': center,
                     'radius': radius
                     },
          'erosion': n_iter_er,
          'dilation': n_iter_dil
          }

# Get train and test data
# Obtain the data
# img_id = 'mpeg7'
# train_images, train_labels, test_images, test_labels, labels_map = utils.get_mpeg_data()

# img_id = 'mnist'
# train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

img_id = 'cifar10'
train_images, train_labels, test_images, test_labels = utils.get_cifar()

N_train = train_images.shape[0]
N_test = test_images.shape[0]
images = np.concatenate([train_images,test_images], axis=0)
pd = persistence_diagram.PDiagram(images, fil_parms=params, man_dim=man_dim, images_id = img_id)
data, num_of_filtrations, num_of_hom, max_num_of_pnts = pd.get_embedded_pds()

x_train = data[:N_train]
x_test = data[N_train:]
y_train = train_labels
y_test = test_labels



# Set up model
in_shape = [num_of_filtrations, num_of_hom, max_num_of_pnts, man_dim]
per_model = model.build_model(input_shape=in_shape, K=K)
print(per_model.summary())
# Setup datasets
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Instantiate metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 30
for epoch in range(epochs):
    tf.print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        x_batch_train = tf.split(x_batch_train, num_or_size_splits=num_of_filtrations, axis=1)
        x_batch_train = [tf.squeeze(_) for _ in x_batch_train]

        with tf.GradientTape() as tape:
            logits = per_model(x_batch_train, training=True)  # Logits for this minibatch
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, per_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, per_model.trainable_weights))
        train_acc_metric(y_batch_train, logits)

        # # Log every 50 batches.
        # if step % 50 == 0:
        #     tf.print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
        #     tf.print('Seen so far: %s samples' % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        x_batch_val = tf.split(x_batch_val, num_or_size_splits=num_of_filtrations, axis=1)
        x_batch_val = [tf.squeeze(_) for _ in x_batch_val]

        val_logits = per_model(x_batch_val)
        # Update val metrics
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: %s' % (float(val_acc),))
