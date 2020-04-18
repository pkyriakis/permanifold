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
K = 20  # number of projection bases

def pad_diagrams(dgms, max_pnts):
    '''
        Adds zeros points to the diagrams to make them of equal size
        and pad the points to bring them to the the man_dim size
    '''
    # Get the highest number of points in a PD
    # and the number of homologies
    max_num_of_pnts = 0
    num_of_dgms = 0
    num_of_hom = 0
    for ind in dgms.keys():
        num_of_dgms = 0
        for filtration in dgms[ind].keys():
            for par_ind in dgms[ind][filtration].keys():
                dgm = dgms[ind][filtration][par_ind]
                max_num_of_pnts = max(max_num_of_pnts, dgm.shape[0])
                num_of_dgms += 1
                num_of_hom = len(set(dgm[:,2]))

    # Pad
    N = len(dgms.keys())
    max_num_of_pnts = max(max_num_of_pnts, max_pnts)
    out = np.zeros([N, num_of_dgms, num_of_hom, max_num_of_pnts, man_dim], dtype=np.float32)
    for ind in dgms.keys():
        cnt = 0
        for filtration in dgms[ind].keys():
            for par_ind in dgms[ind][filtration].keys():
                dgm = dgms[ind][filtration][par_ind]
                for p_ind in range(dgm.shape[0]):
                    hom = int(dgm[p_ind, 2])
                    out[ind, cnt, hom, p_ind, :2] = dgm[p_ind, :2]
                cnt += 1
    return out, num_of_hom, max_num_of_pnts

def post_process(dgms_train, dgms_test):
    '''
        Post processing of the persistance diagrams;
        also find the number of filtrations, homology classes and max number of points across diagrams
    '''
    x_train, num_of_hom, max_train = pad_diagrams(dgms_train, 0)
    x_test, num_of_hom, max_test = pad_diagrams(dgms_test, 0)

    # Train and test need to be padded to same length
    if max_train > max_test:
        x_test, num_of_hom, max_test = pad_diagrams(dgms_test, max_train)
    elif max_test > max_train:
        x_train, num_of_hom, max_train = pad_diagrams(dgms_train, max_test)
    max_num_points = max(max_train,max_test)
    num_of_filtrations = x_test.shape[1]
    return x_train, x_test, num_of_filtrations, num_of_hom, max_num_points



# Obtain the data
img_id = 'mnist'
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()
train_images = train_images.reshape(train_images.shape[0], 28 * 28).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28 * 28).astype('float32')

## Set the params of the filtrations
# Height filtration
num_of_vects = 6
angles = np.linspace(0, math.pi / 2, num_of_vects)
dirs = [[round(math.cos(theta), 2), round(math.sin(theta), 2)] for theta in angles]
dirs = np.array(dirs)

# Radial filtration
center = np.array([[10, 10], [10, 20], [20, 10], [20, 20]])
radius = np.array([5, 10, 15])

# Erosion filtration
n_iter_er = np.array([1, 2, 3, 50])

# Dilation filtration
n_iter_dil = np.array([1, 3, 5, 10, 50])

params = {'cubical': None,
          'height': dirs,
          'radial': {'center': center,
                     'radius': radius
                     },
          'erosion': n_iter_er,
          'dilation': n_iter_dil
          }
# Get persistence diagrams
pd_train = persistence_diagram.PDiagram(train_images, fil_parms=params, images_id='mnist_train')
pd_test = persistence_diagram.PDiagram(test_images, fil_parms=params, images_id='mnist_test')

dgms_train = pd_train.get_pds()
dgms_test = pd_test.get_pds()

# Get train and test data
x_train, x_test, num_of_filtrations, num_of_hom, max_num_of_pnts = post_process(dgms_train, dgms_test)
y_train = train_labels
y_test = test_labels

# Set up model
in_shape = [num_of_filtrations, num_of_hom, max_num_of_pnts, man_dim]
per_model = model.build_model(input_shape=in_shape, K=K)

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


epochs = 3
for epoch in range(epochs):
    tf.print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        x_batch_train = tf.split(x_batch_train, num_or_size_splits=28, axis=1)
        x_batch_train = [tf.squeeze(_) for _ in x_batch_train]

        with tf.GradientTape() as tape:
            logits = per_model(x_batch_train, training=True)  # Logits for this minibatch
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, per_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, per_model.trainable_weights))
        train_acc_metric(y_batch_train, logits)

        # Log every 50 batches.
        if step % 50 == 0:
            tf.print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            tf.print('Seen so far: %s samples' % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        x_batch_val = tf.split(x_batch_val, num_or_size_splits=28, axis=1)
        x_batch_val = [tf.squeeze(_) for _ in x_batch_val]

        val_logits = per_model(x_batch_val)
        # Update val metrics
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: %s' % (float(val_acc),))
