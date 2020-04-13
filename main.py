'''
    Learning Persistent Manifold Representations - NeurIPS Submission
'''

from persistence_diagram import PDiagram
import utils
import numpy as np
# print(np.where(train_labels==8))
# dir = np.array([0,1])
# fil = HeightFiltration(direction=dir)
# xt = fil.transform(train_images)
# #xt = train_images
# #print(fil.mesh_)
# plt.imshow(xt[0])
# ind = np.where(train_labels==9)
# cub = CubicalPersistence(homology_dimensions=(0,1,2,3))
# cub.fit(xt)
# cub.transform_plot(xt,sample=0)

train_images, train_labels, test_images, test_labels = utils.get_mnist_data()

train_images = np.expand_dims(train_images[0],axis=0)

param = {'cubical' : None,
         'height': np.array([[0,1],[1,0]]),
         'radial': np.array([[10,3],[20,6]]),
         'erosion': [5],
         'dilation': [10]
         }
PD = PDiagram(train_images, param, man_dim=81)
pds = PD.get_pds()
vpds = PD.get_vectorized_pds()
# print(pds[0]['cubical'][0])
# print(pds[0]['height'][0])
# print(pds[0]['height'][1])
# print(pds[0]['radial'][0])
# print(pds[0]['radial'][1])
# print(pds[0]['erosion'][0])
# print(pds[0]['dilation'][0])

# print(vpds[0]['cubical'])

for fil in vpds[0].keys():
    for i in vpds[0][fil]:
        print(fil)
        print(pds[0][fil][i])
        print()
        print(vpds[0][fil][i])
        print()