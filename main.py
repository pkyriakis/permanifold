'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''

from persistence_diagram import PDiagram
import utils

train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

pd = PDiagram(train_images[0:10,:,:], images_id= 'mnist_train',save_every=100)
dgms_dict = pd.get_peristence_diagrams()

# pd = PDiagram(test_images, images_id= 'mnist_test',save_every=10)
# dgms_dict = pd.get_peristence_diagrams()
