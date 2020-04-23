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
pd = persistence_diagram.PDiagram(images, fil_parms=params, images_id = img_id)
inputs, num_of_filtrations, num_of_hom, max_num_of_pnts = pd.get_pds()

x_train = []
x_test = []
for input in inputs:
    x_train.append(input[:N_train])
    x_test.append(input[N_train:])
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

# Set up model
in_shape = [num_of_filtrations, num_of_hom, max_num_of_pnts, man_dim]
per_model = model.build_model(input_shape=in_shape, K=K, units=[256,128,10])
print(per_model.summary())

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

per_model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

per_model.fit(x=x_train,
          y=y_train,
          epochs=10,
          batch_size=64,
          callbacks=[tensorboard_callback],
          validation_data=(x_test, y_test))