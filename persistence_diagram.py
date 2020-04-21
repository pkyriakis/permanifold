import math

from tqdm import tqdm

import os
import pickle
import random

from scipy.stats import multivariate_normal
from collections import defaultdict
from time import sleep

import numpy as np
import multiprocessing
from scipy.integrate import simps

from gtda.images import *
from gtda.homology import CubicalPersistence


class PDiagram():
    '''
        Main class that generates PDs
    '''
    def __init__(self, images, fil_parms, images_id='mnist', man_dim=9, embedding = 'padding'):
        self.images = images
        self.save_dir = 'data/diagrams/' + images_id + "/"
        self.fil_params = fil_parms
        self.man_dim = man_dim
        self.embedding = embedding

        self.num_images = images.shape[0]
        self.image_size = images.shape[1]
        self.img_inds = list(range(self.num_images))

        self.manager = multiprocessing.Manager()
        self.dgms_dict = self.manager.dict()
        self.vdgms_dict = self.manager.dict()
        self.edgms_dict = self.manager.dict()
        self.done = multiprocessing.Value("i", 0)
        self.max_num_of_points = multiprocessing.Value("i", 0)
        self.num_of_hom = multiprocessing.Value("i", 0)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __show_progress(self, total):
        '''
            Shows a progress bar
        '''
        while self.done.value < total:
            self.progress_bar.update(self.done.value)
            sleep(1)

    def __chunks(self, lst, n):
        '''
            Yield successive n-sized chunks from lst.
        '''
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def __rotate_point(self, point):
        '''
            Rotates a given point by 45deg to bring it to birth-persinstence form
            @param: point - the point to be rotated
        '''
        birth = point[0]
        death = point[1]
        return [birth, death - birth]

    def __compute_pds_chunk(self, chunk):
        '''
            Computes the PDs for the given chunk of images using the params set in the self.fil_params
        :return:
        '''
        for index in chunk:
            image = np.expand_dims(self.images[index], axis=0)
            bin = Binarizer()
            bin_image = bin.fit_transform(image)
            dgm_image = dict()
            for filtration in self.fil_params.keys():
                params = self.fil_params[filtration]
                cubical = CubicalPersistence(homology_dimensions=(0, 1))
                if filtration == 'cubical':
                    cub_dict = dict()
                    cub_dict[0] = np.squeeze(cubical.fit_transform(image))
                    dgm_image[filtration] = cub_dict
                if filtration == 'height':
                    height_dict = dict()
                    for i in range(params.shape[0]):
                        heigtht_fil = HeightFiltration(direction=params[i])
                        filtered = heigtht_fil.fit_transform(bin_image)
                        filtered = filtered / np.max(filtered)
                        height_dict[i] = np.squeeze(cubical.fit_transform(filtered))
                    dgm_image[filtration] = height_dict
                if filtration == 'radial':
                    radial_dict = dict()
                    center = params['center']
                    radius = params['radius']
                    cnt = 0
                    for i in range(center.shape[0]):
                        for j in range(radius.shape[0]):
                            radial_fil = RadialFiltration(center=center[i], radius=radius[j])
                            filtered = radial_fil.fit_transform(bin_image)
                            filtered = filtered / np.max(filtered)
                            radial_dict[cnt] = np.squeeze(cubical.fit_transform(filtered))
                            cnt += 1
                    dgm_image[filtration] = radial_dict
                if filtration == 'dilation':
                    dil_dict = dict()
                    for i in range(len(params)):
                        dial_fil = DilationFiltration(n_iterations=int(params[i]))
                        filtered = dial_fil.fit_transform(bin_image)
                        filtered = filtered / np.max(filtered)
                        dil_dict[i] = np.squeeze(cubical.fit_transform(filtered))
                    dgm_image[filtration] = dil_dict
                if filtration == 'erosion':
                    er_dict = dict()
                    for i in range(len(params)):
                        er_fil = ErosionFiltration(n_iterations=int(params[i]))
                        filtered = er_fil.fit_transform(bin_image)
                        filtered = filtered / np.max(filtered)
                        er_dict[i] = np.squeeze(cubical.fit_transform(filtered))
                    dgm_image[filtration] = er_dict

                # Store it in dict
                self.dgms_dict[index] = dgm_image
                self.done.value += 1

    def __appr_integral(self, fun, x_low, x_up, y_low, y_up, K=5):
        '''
            Approxiamates the 2d integral of the given function over a K-grid
        '''
        x = np.linspace(x_low, x_up, K)
        y = np.linspace(y_low, y_up, K)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                zz[i, j] = fun(xx[i, j], yy[i, j])
        res = simps(simps(zz, x), y)
        return res

    def __vectorize_pds_chunk(self, chunk):
        '''
            Vectorize the given chunk of PDs
        '''
        cov =  .2*np.eye(2)
        phi = lambda x, y, mu: multivariate_normal(mean=mu, cov=cov).pdf([x, y])
        grid_dim = int(np.sqrt(self.man_dim) + 1)
        for index in chunk:
            dgms = self.dgms_dict[index]
            vdgms = dict()
            for filtration in dgms.keys():
                vdgms_filtration = dict()
                for ind in dgms[filtration].keys():
                    dgm = dgms[filtration][ind]
                    # Get PD for each homology class
                    dgm_of_hom = defaultdict(list)
                    for point in dgm:
                        hom = point[-1]
                        birth, peristence = self.__rotate_point(point[:2])
                        dgm_of_hom[hom].append([birth, peristence])

                    # Init vectorization
                    num_of_hom = len(dgm_of_hom.keys())
                    vectorized_dgm = np.zeros([self.man_dim, num_of_hom])

                    # Iterate over all homology classes
                    for hom in dgm_of_hom.keys():
                        # Get rid of dublicate points to speed up
                        unique_points = defaultdict(int)
                        for point in dgm_of_hom[hom]:
                            tpnt = tuple(point)
                            unique_points[tpnt] = unique_points[tpnt] + 1  # count multiplicity

                        rho = lambda x, y: \
                            sum([mult * phi(x, y, list(mu)) for mu, mult in unique_points.items()])

                        cnt = 0

                        lin_space = np.linspace(0, 1, grid_dim)
                        # Compute vectorization
                        lin_space = np.linspace(0, 1, grid_dim)
                        mx, my = np.meshgrid(lin_space, lin_space)
                        for i in range(grid_dim - 1):
                            for j in range(grid_dim - 1):
                                x_low = mx[i, j]
                                x_up = mx[i, j + 1]
                                y_low = my[i, j]
                                y_up = my[i + 1, j]
                                vectorized_dgm[cnt, int(hom)] = self.__appr_integral(rho, x_low, x_up, y_low, y_up )
                                cnt += 1

                    vdgms_filtration[ind] = vectorized_dgm
                vdgms[filtration] = vdgms_filtration
            # Store it in dict
            self.vdgms_dict[index] = vdgms
            self.done.value += 1

    def __embed_pds_chunk(self, chunk):
        '''
            Embeds points in an Euclidean space (for the given chunk of indices)
        '''
        lin_space = np.linspace(0, 1, int(np.sqrt(self.man_dim)))
        grid_x, grid_y = np.meshgrid(lin_space, lin_space)
        cov = np.eye(2)
        rho = lambda x, y, mu: multivariate_normal(mean=mu, cov=cov).pdf([x, y])
        for index in chunk:
            edgm_image = dict()
            for filtration in self.dgms_dict[index].keys():
                edgm_param = dict()
                for par_ind in self.dgms_dict[index][filtration].keys():
                    dgm = self.dgms_dict[index][filtration][par_ind]

                    # Update counter for common attributes
                    self.max_num_of_points.value = max(self.max_num_of_points.value, dgm.shape[0])
                    self.num_of_hom.value = len(set(dgm[:, -1]))

                    embeded_points = []
                    if self.embedding == 'padding':
                        for point in dgm:
                            hom = point[-1]
                            embeded_point = [0] * (self.man_dim + 1)
                            embeded_point[:2] = self.__rotate_point(point[:2])
                            embeded_point[-1] = hom
                            embeded_points.append(embeded_point)
                    elif self.embedding == 'gauss_grid':
                        for point in dgm:
                            hom = point[-1]
                            birth, peristence = self.__rotate_point(point[:2])
                            embeded_point = []
                            for dim in range(self.man_dim):
                                i = dim // np.sqrt(self.man_dim)
                                j = dim % np.sqrt(self.man_dim)
                                i = int(i)
                                j = int(j)
                                mu = [grid_x[i, j], grid_y[i, j]]
                                embeded_point.append(rho(birth, peristence, mu))
                            embeded_point.append(hom)
                            embeded_points.append(embeded_point)
                    edgm_param[par_ind] = np.array(embeded_points)
                edgm_image[filtration] =  edgm_param
            self.edgms_dict[index] = edgm_image
            self.done.value += 1


    def __run_parallel(self, target, indices):
        '''
            Parellel run of target; the given list of indices creates the chunks to allocate to each cpu
        '''
        cpus = multiprocessing.cpu_count()
        chuck_size = math.ceil(len(indices) / cpus)

        jobs = []
        for chunk in self.__chunks(indices, chuck_size):
            p = multiprocessing.Process(target=target,
                                        args=(chunk,))
            p.start()
            jobs.append(p)

        # total = len(indices)
        # self.progress_bar = tqdm(total=total)
        # bp = multiprocessing.Process(target=self.__show_progress, args=(total,))
        # bp.start()
        #jobs.append(bp)

        for job in jobs:
            job.join()


    def __get_pds(self):
        '''
            Calculates the persistence diagrams of all images
        :return: a dict, keyed by the index of image
        '''

        print('Computing persistence diagrams...')
        self.__run_parallel(self.__compute_pds_chunk, self.img_inds)

        return self.dgms_dict

    def get_vectorized_pds(self):
        '''
            Returns the vectorized version of the PDs using the persistent images method
            @https://arxiv.org/pdf/1507.06217.pdf
        '''
        self.__get_pds()

        # Make sure counter set to zero
        computed_inds = list(self.vdgms_dict.keys())
        inds_left = list(filter(lambda i: i not in computed_inds, self.img_inds))
        self.done.value = len(computed_inds)
        if len(inds_left) > 0:
            print('Vectorizing persistence diagrams...')
            self.__run_parallel(self.__vectorize_pds_chunk, inds_left)

        # # Store to file
        out_file = open(self.save_dir + 'vectorized_dgms.pkl', 'wb')
        pickle.dump([self.fil_params, self.vdgms_dict.copy()], out_file)
        out_file.close()

        return self.vdgms_dict

    def __reformat_pad_diagrams(self, chunk):
        '''
            Converts the self.edgms dict to numpy array and
            pads the diagrams to make them of equal size
        '''
        data = np.ctypeslib.as_array(self.shared_data_array)
        for ind in chunk:
            cnt = 0
            for filtration in self.edgms_dict[ind].keys():
                for par_ind in self.edgms_dict[ind][filtration].keys():
                    dgm = self.edgms_dict[ind][filtration][par_ind]
                    for p_ind in range(dgm.shape[0]):
                        hom = int(dgm[p_ind, -1])
                        data[ind, cnt, hom, p_ind, :] = dgm[p_ind, :self.man_dim]
                    cnt += 1

    def get_embedded_pds(self):
        '''
            Get embedded PDs
        '''
        # Check if already there
        for filename in os.listdir(self.save_dir):
            if ".pkl" in filename:
                with open(os.path.join(self.save_dir,filename), 'rb') as f:
                    [fil_params, man_dim, data] = \
                        pickle.load(f)
                    equal = True
                    if self.num_images != data.shape[0]:
                        equal = False
                    if self.man_dim != man_dim:
                        equal=False
                    if set(self.fil_params.keys()) != set(fil_params.keys()):
                        equal = False
                    for filtration in self.fil_params:
                        vself = self.fil_params[filtration]
                        vin = fil_params[filtration]
                        if type(vin).__module__ == np.__name__:
                            if not np.array_equal(vin,vself):
                                equal = False
                                break
                        if type(vin) is dict:
                            for k in vin.keys():
                                if not np.array_equal(vin[k],vself[k]):
                                    equal = False
                    if equal:
                        return data, data.shape[1], data.shape[2], data.shape[3]

        self.__get_pds()
        assert set(self.dgms_dict.keys()) == set(self.img_inds)

        # Get number of filtrations
        self.num_of_filtrations = 0
        for filtration in self.dgms_dict[0].keys():
            for _ in self.dgms_dict[0][filtration]:
                self.num_of_filtrations += 1

        print('Embedding persistence diagrams...')
        self.__run_parallel(self.__embed_pds_chunk, self.img_inds)

        print('Padding persistence diagrams...')
        self.data = np.ctypeslib.as_ctypes(np.zeros(shape=[self.num_images, self.num_of_filtrations,
                         self.num_of_hom.value, self.max_num_of_points.value, self.man_dim], dtype=np.float32))
        self.shared_data_array = multiprocessing.sharedctypes.RawArray(self.data._type_, self.data)
        self.__run_parallel(self.__reformat_pad_diagrams, self.img_inds)

        data = np.ctypeslib.as_array(self.shared_data_array)

        # # Save to file
        fname = os.path.join(self.save_dir, 'edgms.pkl-' + str(random.randint(0,1000)))
        out_file = open(fname, 'wb')
        pickle.dump([self.fil_params, self.man_dim, data], out_file, protocol=-1)
        out_file.close()

        return np.ctypeslib.as_array(self.shared_data_array), self.num_of_filtrations, \
               self.num_of_hom.value, self.max_num_of_points.value
