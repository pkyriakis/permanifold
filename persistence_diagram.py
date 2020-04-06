import dionysus as dn
import numpy as np
import os
import matplotlib.pyplot as plt
import utils
from datetime import datetime
import pickle



class PDiagram():
    def __init__(self,images, images_id = 'mnist', save_every = 100, num_of_fil = 100):
        self.images = images
        self.save_dir = 'data/dgms/' + images_id + "/"
        self.save_every = save_every
        self.num_images = images.shape[0]
        self.image_size = images.shape[1]
        self.dgms_dict = None
        self.f_values = np.sort(np.random.uniform(size=num_of_fil))

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        else:
            self.__load_pds()


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


    def __load_pds(self):
        '''
            Loads PDs if present in dir
        :return:
        '''
        self.dgms_dict = dict()
        for filename in os.listdir(self.save_dir):
            with open(os.path.join(self.save_dir,filename), 'rb') as f:
                input = np.load(f)
                for key in input.keys():
                    self.dgms_dict[int(key)] = input[key]

        print('Loaded {} persistence diagrams.'.format(len(self.dgms_dict)))


    def __save_pds(self, start, end, counter):
        '''
            Saves the dict of diagrams from start to end (not inclusive)
        :param counter: int to append to filename
        :return:
        '''

        dgms_dict_to_save = dict() # contains an entry for each image/diagram, keyed by the index of the image in self.images
        for n in range(start,end+1):
            dgms_dict_to_save[str(n)] = self.dgms_dict[n]

        np.savez(self.save_dir + "ckpnt_" + str(counter), **dgms_dict_to_save)
        print('Saved checkpoint file from start = {} to end = {}.'.format(start,end))

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

        min_intensity = np.amax(image)
        for simplex in complex:
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

    def __compute_pds_chunk(self, start, end):
        '''
            Computes the PDs for the given chunk of images, start to end
        :param start:
        :param end:
        :return:
        '''

    def get_peristence_diagrams(self):
        '''
            Calculates the peristence diagrams of all images
        :return: a dict of list, keyed by the index of image in the self.images
        '''

        # Check if already loaded
        if self.dgms_dict:
            if len(self.dgms_dict) == self.num_images: # all PDs computed
                return self.dgms_dict
            else: # not all computed; start from the first not computed
                start = max(list(self.dgms_dict.keys())) + 1
        else: # need to initialize
            self.dgms_dict = dict()
            start = 0

        for n in range(start, self.num_images):
            # Get the diagrams of the n-th image
            image = self.images[n,:,:]
            complex = self.__get_complex_from_image(image)
            fil = self.__get_filtation_from_complex(image, complex)
            m = dn.homology_persistence(fil)
            dgms = dn.init_diagrams(m, fil)

            # Convert it to list
            dgm_id = 0
            points = []
            for dgm in dgms:
                for point in dgm:
                    birth = point.birth
                    death = point.death
                    points.append([dgm_id, birth, death])
                dgm_id = dgm_id + 1

            # Store it in dict
            self.dgms_dict[n] = points

            # Save checkpoint to file
            if (n + 1) % self.save_every == 0:
                self.__save_pds(n + 1 - self.save_every, n, (n+1)//self.save_every)
                print('Finished {}/{}'.format(n+1,self.num_images))
        return self.dgms_dict

