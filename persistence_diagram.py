import os
import pickle
import random
import math
import multiprocessing
from collections import defaultdict

import numpy as np
import networkx as nx
import cechmate as cm

from tqdm import tqdm
from scipy.stats import multivariate_normal
from scipy.integrate import simps
from gtda.images import *
from gtda.homology import VietorisRipsPersistence
from gtda.graphs import GraphGeodesicDistance
from gtda.homology import CubicalPersistence


class ImagePDiagram():
    '''
        Main class that generates PDs for images
    '''

    def __init__(self, images, fil_parms, images_id='mnist'):
        self.images = images
        self.images_id = images_id
        self.save_dir = 'diagrams/' + images_id + "/"
        self.fil_params = fil_parms

        self.num_images = images.shape[0]
        self.image_size = images.shape[1]
        self.img_inds = list(range(self.num_images))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __reshape_dgm(self, dgms):
        '''
            Splits diagrams accross homology classes
        '''
        N = dgms.shape[0]
        pnts = dgms.shape[1]
        out = np.zeros(shape=(N, 2, pnts, 2))
        for ind in range(dgms.shape[0]):
            cur0 = dgms[ind].copy()
            cur1 = dgms[ind].copy()
            mask0 = (cur0[:, 2] == np.ones_like(cur0[:, 2]))
            mask1 = (cur1[:, 2] == np.zeros_like(cur1[:, 2]))
            cur0[mask0] = 0
            cur1[mask1] = 0
            out[ind, 0, :, :] = cur0[:, :2]
            out[ind, 1, :, :] = cur1[:, :2]
        return out

    def __get_pds(self):
        '''
            Calculates the persistence diagrams for all images
        :return: a list
        '''
        binarizer = Binarizer(n_jobs=-1)
        bin_image = binarizer.fit_transform(self.images)
        pds = []  # List containing PDs for each filtration
        bar = tqdm(total=0)
        for filtration, params in self.fil_params.items():
            cubical = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=-1)
            if filtration == 'cubical' and params:
                bar.total = 1
                bar.refresh()
                dgms = cubical.fit_transform(self.images)
                pds.append(self.__reshape_dgm(dgms))
                bar.update(1)
            if filtration == 'height':
                bar.total = params.shape[0]
                bar.refresh()
                for i in range(params.shape[0]):
                    heigtht_fil = HeightFiltration(direction=params[i])
                    filtered = heigtht_fil.fit_transform(bin_image)
                    filtered = filtered / np.max(filtered)
                    dgms = cubical.fit_transform(filtered)
                    pds.append(self.__reshape_dgm(dgms))
                    bar.update(1)
            if filtration == 'radial':
                center = params['center']
                radius = params['radius']
                cnt = 0
                bar.total = center.shape[0] * radius.shape[0]
                bar.refresh()
                for i in range(center.shape[0]):
                    for j in range(radius.shape[0]):
                        radial_fil = RadialFiltration(center=center[i], radius=radius[j])
                        filtered = radial_fil.fit_transform(bin_image)
                        filtered = filtered / np.max(filtered)
                        dgms = cubical.fit_transform(filtered)
                        pds.append(self.__reshape_dgm(dgms))
                        cnt += 1
                        bar.update(1)
            if filtration == 'dilation':
                bar.total = len(params)
                bar.refresh()
                for i in range(len(params)):
                    dial_fil = DilationFiltration(n_iterations=int(params[i]))
                    filtered = dial_fil.fit_transform(bin_image)
                    filtered = filtered / np.max(filtered)
                    dgms = cubical.fit_transform(filtered)
                    pds.append(self.__reshape_dgm(dgms))
                    bar.update(1)
            if filtration == 'erosion':
                bar.total = len(params)
                bar.refresh()
                for i in range(len(params)):
                    er_fil = ErosionFiltration(n_iterations=int(params[i]))
                    filtered = er_fil.fit_transform(bin_image)
                    filtered = filtered / np.max(filtered)
                    dgms = cubical.fit_transform(filtered)
                    pds.append(self.__reshape_dgm(dgms))
                    bar.update(1)
        return pds

    def get_pds(self):
        '''
            Get embedded PDs
        '''
        # Check if already there
        if self.images_id != 'mpeg7':
            for filename in os.listdir(self.save_dir):
                if ".pkl" in filename:
                    with open(os.path.join(self.save_dir, filename), 'rb') as f:
                        [fil_params, data] = pickle.load(f)
                        equal = True

                        # Need to make sure that stored PD is generated using
                        # the same filtration params as the ones provided
                        if self.num_images != data[0].shape[0]:
                            equal = False
                        if set(self.fil_params.keys()) != set(fil_params.keys()):
                            equal = False
                        for filtration in self.fil_params:
                            vself = self.fil_params[filtration]
                            vin = fil_params[filtration]
                            if type(vin).__module__ == np.__name__:
                                if not np.array_equal(vin, vself):
                                    equal = False
                                    break
                            if type(vin) is dict:
                                for k in vin.keys():
                                    if not np.array_equal(vin[k], vself[k]):
                                        equal = False
                        if equal:
                            print('Loaded persistence diagrams.')
                            return data

        print('Computing persistence diagrams.')
        pds = self.__get_pds()

        # # Save to file
        if self.images_id != 'mpeg7':
            fname = os.path.join(self.save_dir, 'dgms.pkl-' + str(random.randint(0, 1000)))
            out_file = open(fname, 'wb')
            pickle.dump([self.fil_params, pds], out_file, protocol=-1)
            out_file.close()

        return pds


class GraphPDiagram():
    '''
        Main class the computes PDs for graphs
    '''

    def __init__(self, graphs, graphs_id, filtrations=None):
        if filtrations is None:
            filtrations = ['vr', 'degree', 'avg_path']

        self.graphs = graphs
        self.graphs_id = graphs_id
        self.filtrations = filtrations
        self.save_dir = 'diagrams/' + graphs_id
        self.distances = None
        self.dgms = multiprocessing.Manager().dict()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def set_graphs(self, graphs):
        self.graphs = graphs

    def __chunks(self, lst, n):
        '''
            Yield successive n-sized chunks from lst.
        '''
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def save_pds(self, x_train, y_train, x_test, y_test):
        '''
            Save train/test PDs and labels
        '''
        fname = os.path.join(self.save_dir, 'dgms.pkl-' + str(random.randint(0, 1000)))
        out_file = open(fname, 'wb')
        pickle.dump([x_train, y_train, x_test, y_test, self.filtrations],
                    out_file, protocol=-1)
        out_file.close()

    def load_pds(self):
        '''
            Tries load PDs if already computed
        '''
        for filename in os.listdir(self.save_dir):
            if ".pkl" in filename:
                with open(os.path.join(self.save_dir, filename), 'rb') as f:
                    [x_train, y_train, x_test, y_test,filtrations] = pickle.load(f)
                    if set(filtrations) == set(self.filtrations):
                        print('Loaded persistence diagrams.')
                        return x_train, y_train, x_test, y_test
        return None, None, None, None

    def get_pds(self):
        '''
            Get persistence diagrams
        '''
        pds = []
        if 'vr' in self.filtrations:
            print('Computing Vietoris Rips Graph Persistence')
            pds.append(self.__compute_vr_persistence())
        if 'degree' in self.filtrations:
            print('Computing Lower-Star Persistence using node degree')
            pds.append(self.__get_lower_star_parallel('degree'))
        if 'avg_path' in self.filtrations:
            print('Computing Lower-Star Persistence using average path length')
            pds.append(self.__get_lower_star_parallel('avg_path'))

        return pds

    def __set_distance_matrix(self):
        '''
            Computes and stores the distance matrix for all graphs
        '''
        list_adj = []

        # Convert to list of np.array adj martices
        for graph in self.graphs:
            adj = nx.convert_matrix.to_numpy_array(graph)
            list_adj.append(adj)

        # Get distances
        distances = GraphGeodesicDistance(n_jobs=-1).\
            fit_transform(list_adj)
        self.distances = list(distances)

    def __compute_vr_persistence(self):
        '''
            Computes the Vietoris Rips Persistence
        '''

        # Get distance if not there
        if self.distances is None:
            self.__set_distance_matrix()

        # Compute max distance
        max_dist = []
        for ind in range(len(self.distances)):
            dist = np.max(self.distances[ind])
            max_dist.append(dist)

        dgms = VietorisRipsPersistence(metric='precomputed',
                                     n_jobs=-1).fit_transform(self.distances)

        # Replace inf value with max distance in corresponding graph
        clipped_dgms = []
        for dgm, m_dist in zip(dgms, max_dist):
            dgm[dgm == np.inf] = m_dist
            clipped_dgms.append(dgm)
        clipped_dgms = np.array(clipped_dgms)

        return self.__reshape_dgm(clipped_dgms)

    def __reshape_dgm(self, dgms):
        '''
            Splits diagrams accross homology classes
        '''
        N = dgms.shape[0]
        pnts = dgms.shape[1]
        out = np.zeros(shape=(N, 2, pnts, 2))
        for ind in range(dgms.shape[0]):
            cur0 = dgms[ind].copy()
            cur1 = dgms[ind].copy()
            mask0 = (cur0[:, 2] == np.ones_like(cur0[:, 2]))
            mask1 = (cur1[:, 2] == np.zeros_like(cur1[:, 2]))
            cur0[mask0] = 0
            cur1[mask1] = 0
            out[ind, 0, :, :] = cur0[:, :2]
            out[ind, 1, :, :] = cur1[:, :2]
        return out

    def __get_filtration_values(self, ind, sublevel):
        '''
            Returns the sorted filtration values for the given sublevel set function
        '''
        graph = self.graphs[ind]
        if sublevel == 'degree':
            degrees = sorted([d for (n, d) in graph.degree()])
            degrees = list(set(degrees))
            return degrees
        if sublevel == 'avg_path':
            dist = self.distances[ind]
            means = np.mean(dist, axis=0)
            return sorted(means)

    def __check_sublevel(self, ind, u, val, sublevel):
        '''
            Checks if given node is bellow the given filtration value
        '''
        graph = self.graphs[ind]
        if sublevel == 'degree':
            return graph.degree(u) <= val
        if sublevel == 'avg_path':
            dist = self.distances[ind]
            val_u = np.mean(dist[u,:])
            return val_u <= val

    def __compute_lower_star_persistence(self, sublevel, chunk):
        '''
            Compute PDs by creating a complex using the given sublevel function
        '''

        # Iterate over the given chunk
        for ind in tqdm(chunk):
            graph = self.graphs[ind]
            values = self.__get_filtration_values(ind, sublevel)
            filtration = []
            subgraphs = []

            # Get filtration
            for val in values:
                subgraph = []
                for edge in graph.edges:
                    u, v = edge
                    if self.__check_sublevel(ind, u, val, sublevel):
                        filtration.append(([u], val))
                        subgraph.append(u)
                    if self.__check_sublevel(ind, v, val, sublevel):
                        filtration.append(([v], val))
                        subgraph.append(v)
                    if self.__check_sublevel(ind, u, val, sublevel) \
                            and self.__check_sublevel(ind, v, val, sublevel):
                        filtration.append(([u, v], val))
                subgraphs.append(subgraph)

            # Get persistence diagram
            dgms = cm.phat_diagrams(filtration, show_inf=True, verbose=False)
            # Replace inf values
            new_dgms = []
            for dgm in dgms:
                dgm[dgm == np.inf] = values[-1]
                if dgm.shape[0] > 1000: # few PDs might have an extremely high number of points
                    dgm = dgm[:1000]
                new_dgms.append(dgm)
            dgms = new_dgms

            # PHAT might return only one diagram is there's no H_1 classes
            # add an empty PD cuz the rest of the code expects two diagrams
            if len(dgms) == 1:
                dgms.append(np.array([0, 0]))

            # Store
            self.dgms[ind] = dgms

    def __post_process_dgms(self):
        '''
            Post-processes diagrams after parallel computation is done
        '''
        max_num_of_points = 0
        for ind, _ in enumerate(self.graphs):
            dgm0 = self.dgms[ind][0]
            dgm1 = self.dgms[ind][1]
            new_pnts = max(dgm0.shape[0], dgm1.shape[0])
            max_num_of_points = max(max_num_of_points, new_pnts)

        # Convert to np.array
        N = len(self.dgms.keys())
        out = np.zeros(shape=(N, 2, max_num_of_points, 2))
        for ind, _ in enumerate(self.graphs):
            dgm0 = self.dgms[ind][0]
            out[ind, 0, :dgm0.shape[0], :] = dgm0
            dgm1 = self.dgms[ind][1]
            out[ind, 1, :dgm1.shape[0], :] = dgm1
        return out


    def __get_lower_star_parallel(self, sublevel):
        '''
            Parellel run of target; the given list of indices creates the chunks to allocate to each cpu
        '''

        if sublevel == 'avg_path':
            # Get distance if not there
            if self.distances is None:
                self.__set_distance_matrix()

        indices = range(len(self.graphs))
        cpus = multiprocessing.cpu_count()
        chuck_size = math.ceil(len(indices) / cpus)

        target = self.__compute_lower_star_persistence
        jobs = []
        for chunk in self.__chunks(indices, chuck_size):
            p = multiprocessing.Process(target=target,
                                        args=(sublevel, chunk))
            p.start()
            jobs.append(p)
        for job in jobs:
            job.join()

        diagrams = self.__post_process_dgms()

        # Reset shared vars
        self.dgms = multiprocessing.Manager().dict()

        return diagrams