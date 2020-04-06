import tensorflow as tf

def get_mnist_data(binirize = False):
    '''
        Uses the keras backend to downlaoad and binirize the MNIST images
    :return: train_images, train_labels, test_images, test_labels
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
    if binirize:
        train_images[train_images >= .5] = 1.
        train_images[train_images < .5] = 0.
        test_images[test_images >= .5] = 1.
        test_images[test_images < .5] = 0.
    return train_images, train_labels, test_images, test_labels

def plot_imgage(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

