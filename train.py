import datetime
import os, math

import persistence_diagram
import tensorflow as tf
import layers
import numpy as np
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
man_dim = 2 # Dimension of the manifold
K = 28 # number of projection bases
num_of_hom = 2 # number of homology classes; hardcoded for now (we know its 3 for images);
               # TODO code function to find it

batch_size = 512
epochs = 3
save_every = 1

def pad_diagrams(dgms):
    '''
        Adds zeros points to the diagrams to make them of equal size; tensorflow cant handle inputs with varied size
    '''
    # Get the highest number of points in a PD
    max_num_of_pnts = 0
    for ind in dgms.keys():
        num_of_dgms = 0
        for filtration in dgms[ind].keys():
            for par_ind in dgms[ind][filtration].keys():
                dgm = dgms[ind][filtration][par_ind]
                max_num_of_pnts = max(max_num_of_pnts, dgm.shape[0])
                num_of_dgms += 1

    # Pad
    max_num_of_pnts = 77
    N = len(dgms.keys())
    out = np.zeros([N,num_of_dgms,max_num_of_pnts,man_dim+1], dtype=np.float32)
    for ind in dgms.keys():
        cnt = 0
        for filtration in dgms[ind].keys():
            for par_ind in dgms[ind][filtration].keys():
                dgm = dgms[ind][filtration][par_ind]
                out[ind,cnt,:dgm.shape[0],:] = dgm
                cnt += 1
    return out


# Obtain the data
img_id = 'mnist'
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()
train_images = train_images.reshape(train_images.shape[0], 28*28).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28*28).astype('float32')

## Set the params of the filtrations
# Height filtration
num_of_vects = 6
angles = np.linspace(0, math.pi/2, num_of_vects)
dirs = [[round(math.cos(theta),2),round(math.sin(theta),2)] for theta in angles]
dirs = np.array(dirs)

# Radial filtration
center = np.array([[10,10], [10,20], [20,10], [20,20]])
radius = np.array([5, 10, 15])

# Erosion filtration
n_iter_er = np.array([1,2,3,50])

# Dilation filtration
n_iter_dil = np.array([1,3,5,10,50])

params = {'cubical' : None,
         'height': dirs,
         'radial': {'center' : center,
                    'radius' : radius
                    },
         'erosion': n_iter_er,
         'dilation': n_iter_dil
         }
# Get persistence diagrams
pd_train = persistence_diagram.PDiagram(train_images, fil_parms=params, images_id='mnist_train')
pd_test = persistence_diagram.PDiagram(test_images, fil_parms=params, images_id='mnist_test')

# Get train test data
dgms_train = pd_train.get_pds()
print(len(dgms_train.keys()))
dgms_test = pd_test.get_pds()
x_train = pad_diagrams(dgms_train)
x_test = pad_diagrams(dgms_test)
y_train = train_labels
y_test = test_labels
# x_train = x_train.reshape(60000,77*28*3)
# x_test = x_test.reshape(10000,77*28*3)
#
# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Set up model
per_model = layers.PManifoldModel(K, man_dim, num_of_hom)

# Instantiate an optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Instantiate performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Set up logs
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + img_id + '/' + current_time + '/train'
test_log_dir = 'logs/' + img_id + '/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# # Set up checkpoints
# checkpoint_directory = ".tmp/training_checkpoints/" + img_id
# checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
#                                  optimizer=optimizer, model=per_model)
# manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

# Load weights if any
# checkpoint.restore(manager.latest_checkpoint)
# if manager.latest_checkpoint:
#     print("Restored from {}".format(manager.latest_checkpoint))
# else:
#     print("Training from scratch.")

# inputs = tf.keras.Input(shape=(28*77*3,), name='digits')
# x = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(inputs)
# x = tf.keras.layers.Dense(256, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
# outputs = tf.keras.layers.Dense(10, name='predictions')(x)
#
# model = tf.keras.Model(inputs=[inputs], outputs=outputs)

per_model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
               # Loss function to minimize
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               # List of metrics to monitor
               metrics=['sparse_categorical_accuracy']
               )
#
# print('# Fit model on training data')
history = per_model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=10,
                    validation_data=(x_test,y_test)
                    )
#
# print('\nhistory dict:', history.history)

# Train and test




