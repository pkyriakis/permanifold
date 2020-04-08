'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''

from persistence_diagram import PDiagram
import utils
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

#pd = PDiagram(train_images, images_id= 'mnist_train')
#dgms_dict = pd.get_vectorized_pds()


pd = PDiagram(test_images[0:1,:,:], images_id= 'mnist_test')
embpnts = pd.get_embedded_pds(man_dim=9)
print(embpnts[1])
