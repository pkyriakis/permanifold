import persistence_diagram
import tensorflow as tf
import layers
import numpy as np
import utils

man_dim = 9 # Dimension of the manifold

def pad_diagrams(dgms):
    '''
        Adds zeros points to the diagrams to make the of equal size; tensorflow cant handle inputs with varied size
    '''
    # Get the highest number of points in a PD
    max_num_of_pnts = 0
    for ind in dgms.keys():
        dgm = dgms[ind]
        max_num_of_pnts = max(max_num_of_pnts, dgm.shape[0])

    # Pad
    x = []
    for ind in dgms.keys():
        dgm = dgms[ind]
        x_padded = np.zeros((max_num_of_pnts, man_dim+1)) # plus one cus the first item of each poin is the homology class
        x_padded[:dgm.shape[0],:] = dgm
        x.append(x_padded)
    x_train=np.array(x)

    return x_train

# Obtain the data
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

# Get persistence diagrams
pd = persistence_diagram.PDiagram(test_images[0:10,:,:], images_id = 'mnist_test')
dgms = pd.get_embedded_pds()

x_train = pad_diagrams(dgms)
y_train = test_labels[0:10]

print('sdf')
pm = layers.PManifoldModel(20, 9, 3)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)