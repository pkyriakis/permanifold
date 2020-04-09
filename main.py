'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''

from persistence_diagram import PDiagram
import utils
import numpy as np
train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

#pd = PDiagram(train_images, images_id= 'mnist_train')
#dgms_dict = pd.get_vectorized_pds()


pd = PDiagram(test_images[0:10,:,:], images_id= 'mnist_test')
embpnts = pd.get_embedded_pds()
print(embpnts[0].shape)
# for x in np.nditer(embpnts[0], flags = ['refs_ok']):
#     for y in np.nditer(x, flags = ['refs_ok']):
#         print(y.shape)