import math

import os
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from collections import defaultdict
from time import sleep

import dionysus as dn
import numpy as np
import multiprocessing, random
from scipy.integrate import simps, dblquad


class PDiagram():
    def __init__(self,images, images_id = 'mnist', num_of_fil = 500, man_dim = 9):
        self.images = images
        self.dgms_dir = 'data/dgms/' + images_id + "/"
        self.vdgms_dir = 'data/vdgms/' + images_id + "/"
        self.edgms_dir = 'data/edgms/' + images_id + "/"

        self.f_values = np.linspace(0, 1, num_of_fil)
        self.man_dim = man_dim

        self.num_images = images.shape[0]
        self.image_size = images.shape[1]
        self.img_inds = list(range(self.num_images))

        self.manager = multiprocessing.Manager()
        self.dgms_dict = self.manager.dict()
        self.vdgms_dict = self.manager.dict()
        self.edgms_dict = self.manager.dict()

        if not os.path.exists(self.dgms_dir):
            os.makedirs(self.dgms_dir)
            os.makedirs(self.dgms_dir)
            self.done = multiprocessing.Value("i", 0)
        else:
            self.dgms_dict = self.__load_pds(self.dgms_dir)
            self.done = multiprocessing.Value("i", len(self.dgms_dict))

        if not os.path.exists(self.vdgms_dir):
            os.makedirs(self.vdgms_dir)
        else:
            self.vdgms_dict = self.__load_pds(self.vdgms_dir)

        if not os.path.exists(self.edgms_dir):
            os.makedirs(self.edgms_dir)
        else:
            self.edgms_dict = self.__load_pds(self.edgms_dir)

    def __update_progress(self):
        '''
            Prints a progress
        '''
        while self.done.value <= self.num_images:
            print('Done {}/{}'.format(self.done.value,self.num_images), end="\r")
            if self.done.value < self.num_images:
                sleep(10)
            else:
                return

    def __load_pds(self, dir):
        '''
            Loads if present in dir
        :return:
        '''
        dct = dict()
        for filename in os.listdir(dir):
            with open(os.path.join(dir,filename), 'rb') as f:
                input = np.load(f, allow_pickle=True)
                for key in input.keys():
                    dct[int(key)] = input[key]
        return dct


    def __save_pds(self, dct, dir, chunk):
        '''
            Saves the given dict from start to end (not inclusive)
        :return:
        '''
        dgms_dict_to_save = dict() # contains an entry for each image/diagram, keyed by the index of the image in self.images
        for index in chunk:
            dgms_dict_to_save[str(index)] = dct[index]
        rnd = random.randint(0,10000000)
        np.savez(dir + "ckpnt_" + str(rnd), **dgms_dict_to_save)


    def __chunks(self, lst, n):
        '''
            Yield successive n-sized chunks from lst.
        '''
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


    def __pixel_to_vertices(self,i,j):
        '''
            Given the pixel it returns the vertices' ids of all 4 corner points
        :param i: int
        :param j: int
        :return: list of ints
        '''
        points = [[i, j], [i + 1, j], [i, j + 1], [i + 1, j + 1]]
        vertices = [k * (self.image_size + 1) + l for k, l in points]
        return  vertices

    def __vertex_to_pixels(self, vertex):
        '''
            Inverse of the above, given the vertex id it returns all the adjecent pixels
        :param vertex:
        :return:
        '''
        i = vertex // (self.image_size + 1)
        j = vertex % (self.image_size + 1)

        valid = lambda i,j : i in range(self.image_size) and j in range(self.image_size)
        pixels = []
        if valid(i,j):
            pixels.append([i,j])
        if valid(i-1,j):
            pixels.append([i-1,j])
        if valid(i,j-1):
            pixels.append([i,j-1])
        if valid(i-1,j-1):
            pixels.append([i-1,j-1])
        return pixels


    def __rotate_point(self, point):
        '''
            Rotates a given point by 45deg to bring it to birth-persinstence form and scales down inf deaths if necessary
            @param: point - the point to be rotated
        '''
        birth = point[0]
        death = point[1]
        death = np.min([death, np.max(self.f_values)])

        return [birth, death - birth]


    def __get_complex_from_image(self, image):
        '''
            Obtains a simplicial complex from a given image
        :param image: NxN @numpy.array
        :return: simplicial complex of image: @dionysus.Simplex()
        '''
        complex = []
        for i in range(self.image_size):
            for j in range(self.image_size):
                vertices = self.__pixel_to_vertices(i,j)
                simplex = dn.Simplex(vertices)
                complex.append(simplex)

        return dn.closure(complex,2)


    def __get_filtation_from_complex(self, image, complex):
        '''
            Assigns a real value to each simplex in the given complex
            and returns the resulting sublevel set filtration
        :param image:
        :param complex:
        :return:
        '''
        max_int = np.amax(image)
        for simplex in complex:
            min_intensity = max_int
            list_vert = list(simplex)
            pixels_all = []
            for vertex in list_vert:
                pixels_ver = []
                for pixel in self.__vertex_to_pixels(vertex):
                    pixels_ver.append(tuple(pixel))
                pixels_all.append(pixels_ver)
            pixels = set(pixels_all[0])
            for pixel in pixels_all:
                pixels.intersection(set(pixel))
            for pixel in pixels:
                i,j = list(pixel)
                min_intensity = min(min_intensity, image[i,j])
            simplex.data = min_intensity

        filtration = dn.Filtration()
        for value in self.f_values:
            for simplex in complex:
                if simplex.data <= value:
                    filtration.append(simplex)
        return filtration

    def __compute_pds_chunk(self, chunk):
        '''
            Computes the PDs for the given chunk of images, start to end
        :param start: int, starting index
        :param end: int, last index
        :return:
        '''
        for index in chunk:
            # Get the diagrams of the n-th image
            image = self.images[index,:,:]
            complex = self.__get_complex_from_image(image)
            filtration = self.__get_filtation_from_complex(image, complex)
            m = dn.homology_persistence(filtration)
            dgms = dn.init_diagrams(m, filtration)

            # Convert it to list
            dgm_id = 0
            points = []
            for dgm in dgms:
                for point in dgm:
                    birth = point.birth
                    death = point.death
                    death = np.min([death,np.max(self.f_values)])
                    points.append([dgm_id, birth, death])
                dgm_id = dgm_id + 1

            # Store it in dict
            self.dgms_dict[index] = points
            self.done.value += 1

        self.__save_pds(self.dgms_dict, self.dgms_dir, chunk)


    def __appr_integral(self,fun, x_low, x_up, y_low, y_up, K = 3):
        '''
            Approxiamates the 2d integral of the given function over a K-grid
        '''
        x = np.linspace(x_low,x_up,K)
        y = np.linspace(y_low,y_up,K)
        xx, yy = np.meshgrid(x,y)
        zz = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                zz[i,j] = fun(xx[i,j],yy[i,j])
        res = simps(simps(zz,x),y)
        return res


    def __vectorize_pds_chunk(self, chunk):
        '''
            Vectorize the given chunk of PDs
        '''
        cov = 0.1 * np.eye(2)
        phi = lambda x, y, mu: multivariate_normal(mean=mu, cov=cov).pdf([x, y])
        grid_dim = int(np.sqrt(self.man_dim))

        for index in chunk:
            dgm = self.dgms_dict[index]
            # Get PD for each homology class
            dgm_of_hom = defaultdict(list)
            for point in dgm:
                hom = point[0]
                birth, peristence = self.__rotate_point(point[1:])
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
                x_low = lin_space[0]
                y_low = lin_space[0]

                # Compute vectorization
                for i in range(grid_dim):
                    for j in range(grid_dim):
                        x_up = lin_space[i]
                        y_up = lin_space[j]
                        vectorized_dgm[cnt, int(hom)] = rho(x_low, y_low) + rho(x_low, y_up) + rho(x_up, y_low) + rho(x_up, y_up)
                        #integral = self.__appr_integral(rho, x_low, x_up, y_low, y_up)
                        #integral, error = dblquad(rho, y_low, y_up, lambda x : x_low, lambda y : x_up, epsrel=0.01)
                        #vectorized_dgm[cnt, int(hom)] = integral
                        x_low = x_up
                        y_low = y_up
                        cnt += 1

            # Store it in dict
            self.vdgms_dict[index] = vectorized_dgm
            self.done.value += 1
        self.__save_pds(self.vdgms_dict, self.vdgms_dir, chunk)


    def __embed_pds_chunk(self, chunk):
        '''
            Embeds points in PDs in an Euclidean space (for the given chunk of indices)
        '''

        self.get_pds()
        lin_space = np.linspace(0,1,int(np.sqrt(self.man_dim)))
        grid_x,grid_y = np.meshgrid(lin_space,lin_space)
        cov = np.eye(2)
        rho = lambda x, y, mu: multivariate_normal(mean=mu, cov=cov).pdf([x, y])
        for index in chunk:
            dgm = self.dgms_dict[index]
            embeded_points = []  # the 0-th element of each point is its homology class
            # Keep track of unique points to speed up
            unique_points = defaultdict(int)
            for point in dgm:
                if not tuple(point) in unique_points:
                    unique_points[tuple(point)] = 0 # need to figure out how to handle multiplicities
                    hom = point[0]
                    birth, peristence = self.__rotate_point(point[1:])
                    embeded_point = []
                    for dim in range(self.man_dim):
                        i = dim // np.sqrt(self.man_dim)
                        j = dim % np.sqrt(self.man_dim)
                        i = int(i)
                        j = int(j)
                        mu = [grid_x[i, j], grid_y[i, j]]
                        embeded_point.append(rho(birth, peristence, mu))
                    embeded_points.append([hom] + embeded_point)
                else:
                    unique_points[tuple(point)] += 1
            self.done.value += 1
            self.edgms_dict[index] = np.array(embeded_points)
        self.__save_pds(self.edgms_dict, self.edgms_dir, chunk)

    def __run_parallel(self, target, indices):
        '''
            Parellel run of target; the given list of indices creates the chunks to allocate to each cpu
        '''
        cpus = multiprocessing.cpu_count() - 1 # leave one cpu free
        chuck_size = math.ceil(len(indices)/cpus)

        jobs = []
        for chunk in self.__chunks(indices, chuck_size):
            p = multiprocessing.Process(target=target,
                                        args=(chunk,))
            p.start()
            jobs.append(p)

        # Start a job to update progress bar
        p = multiprocessing.Process(target=self.__update_progress)
        jobs.append(p)
        p.start()

        for job in jobs:
            job.join()

    def get_pds(self):
        '''
            Calculates the peristence diagrams of all images
        :return: a dict of lists, keyed by the index of image
        '''

        computed_inds = list(self.dgms_dict.keys())
        inds_left = list(filter(lambda i: i not in computed_inds, self.img_inds))

        if len(inds_left) > 0:
            print('Computing persistence diagrams...')
            self.__run_parallel(self.__compute_pds_chunk, inds_left)

        return self.dgms_dict


    def get_vectorized_pds(self):
        '''
            Returns the vectorized version of the PDs using the persistent images method
            @https://arxiv.org/pdf/1507.06217.pdf
        '''
        self.get_pds()

        # Make sure counter set to zero
        computed_inds = list(self.vdgms_dict.keys())
        inds_left = list(filter(lambda i: i not in computed_inds, self.img_inds))
        self.done.value = len(computed_inds)
        if len(inds_left) > 0:
            print('Vectorizing persistence diagrams...')
            self.__run_parallel(self.__vectorize_pds_chunk, inds_left)
        return self.vdgms_dict

    def get_embedded_pds(self):
        '''
            Get embedded PDs
        '''
        self.get_pds()

        computed_inds = list(self.edgms_dict.keys())
        inds_left = list(filter(lambda i: i not in computed_inds, self.img_inds))
        self.done.value = len(computed_inds)
        if len(inds_left) > 0:
            print('Embedding persistence diagrams...')
            self.__run_parallel(self.__embed_pds_chunk, inds_left)
        return self.edgms_dict










