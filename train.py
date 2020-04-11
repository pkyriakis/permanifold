import datetime
import os

import persistence_diagram
import tensorflow as tf
import layers
import numpy as np
import utils

man_dim = 9 # Dimension of the manifold
K = 20 # number of projection bases
num_of_hom = 3 # number of homology classes; hardcoded for now (we know its 3 for images);
               # TODO code function to find it

batch_size = 16
epochs = 3
save_every = 1

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
    return np.array(x)


# Obtain the data
img_id = 'mnist'
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()
train_images = train_images[:3000]
train_labels = train_labels[:3000]
test_images = test_images[:500]
test_labels = test_labels[:500]

# Get persistence diagrams
pd_train = persistence_diagram.PDiagram(train_images, images_id = img_id + '_train')
pd_test = persistence_diagram.PDiagram(test_images, images_id = img_id + '_test')

# Get train test data
dgms_train = pd_train.get_embedded_pds()
dgms_test = pd_test.get_embedded_pds()
x_train = pad_diagrams(dgms_train)
x_test = pad_diagrams(dgms_test)
y_train = train_labels
y_test = test_labels

# Create TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Set up model
per_model = layers.PManifoldModel(K, man_dim, num_of_hom)

# Instantiate an optimizer and loss
optimizer = tf.keras.optimizers.Adam()
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

# Set up checkpoints
checkpoint_directory = ".tmp/training_checkpoints/" + img_id
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=optimizer, model=per_model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

# Load weights if any
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Training from scratch.")

# Train and test
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))

  # Training step
  for step, (x_train, y_train) in enumerate(train_dataset):
    # Open Gradient tape
    with tf.GradientTape() as tape:
        logits = per_model(x_train, training=True)
        loss = loss_fn(y_train, logits)
    grads = tape.gradient(loss, per_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, per_model.trainable_variables))
    train_loss(loss)
    train_accuracy(y_train,logits)

    # Save logs
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    # Save checkpoint
    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % save_every == 0:
        manager.save()
        print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
        print('Seen so far: %s samples' % ((step + 1) * batch_size))

  # Test step
  for step, (x_test, y_test) in enumerate(test_dataset):
      logits = per_model(x_test, training=False)
      loss = loss_fn(y_test, logits)
      test_loss(loss)
      test_accuracy(y_test, logits)

      # Save logs
      with test_summary_writer.as_default():
          tf.summary.scalar('loss', test_loss.result(), step=epoch)
          tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

      # Print every 100 batches.
      if step % save_every == 0:
          print('Test loss (for one batch) at step %s: %s' % (step, float(loss)))
          print('Seen so far: %s samples' % ((step + 1) * batch_size))

      # Reset metrics every epoch
      train_loss.reset_states()
      test_loss.reset_states()
      train_accuracy.reset_states()
      test_accuracy.reset_states()



