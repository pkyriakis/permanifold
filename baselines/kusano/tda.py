from matplotlib import pyplot as plt
import os
import numpy as np


def mkdir_os(name_dir_save):
    if not os.path.exists(name_dir_save):
        os.mkdir(name_dir_save)


def make_list_pd(name_dir_pd, num_pd, dim_pd, scale=True):
    list_pd = []
    for k in range(num_pd):
        mat_pd = np.loadtxt("%s/pcd_pd/dim%s_%s.txt" % (
            name_dir_pd, dim_pd, k)).reshape(-1, 2)
        # "../torus/pcd3_sample500_num40/pcd_pd/dim1_2.txt"
        if scale:  # scaling to (b^2,d^2) to (b,d)
            list_pd.append(np.sqrt(mat_pd))
        else:  # CGAL uses (b^2,d^2)-coordinate as default
            list_pd.append(mat_pd)
    return list_pd


def norm_pd(list_pd, p=1):
    num_pd = len(list_pd)
    vec_norm = np.empty(num_pd)
    for i in range(num_pd):
        if p == 0:
            vec_norm[i] = list_pd[i].shape[0]
        else:
            vec_pers = list_pd[i][:, 1] - list_pd[i][:, 0]
            vec_norm[i] = np.linalg.norm(vec_pers, p)
    return vec_norm


def parameter_birth_death_pers(list_pd):
    num_pd = len(list_pd)
    vec_b = np.empty(num_pd)
    vec_d = np.empty(num_pd)
    vec_max_p = np.empty(num_pd)
    for k in range(num_pd):
        mat_pd = list_pd[k]
        vec_birth = mat_pd[:, 0]
        vec_death = mat_pd[:, 1]
        vec_pers = vec_death - vec_birth
        vec_b[k] = np.min(vec_birth)
        vec_d[k] = np.max(vec_death)
        vec_max_p[k] = np.max(vec_pers)
    return np.min(vec_b), np.max(vec_d), np.max(vec_max_p)


def parameter_sigma(list_pd, name_save=None, option=""):
    def _sigma(_mat_pd):
        num_points = _mat_pd.shape[0]
        if num_points > 1:
            _vec = np.empty(int(num_points * (num_points - 1) / 2))
            idx_temp = 0
            for _i in range(num_points):
                for _j in range(_i):
                    _vec[idx_temp] = np.linalg.norm(
                        _mat_pd[_i, :] - _mat_pd[_j, :])
                    idx_temp += 1
            _val_sigma = np.median(_vec)
        else:
            _val_sigma = 0
        return _val_sigma

    name_sigma = "%s/sigma.txt" % name_save
    if option == "import" and os.path.exists(name_sigma):
        print("importing median of sigma for PDs")
        vec_sigma = np.loadtxt(name_sigma)
    else:
        num_pd = len(list_pd)
        vec_sigma = np.empty(num_pd)
        #print("computing median of sigma for %s PDs" % num_pd)
        for i in range(num_pd):
            vec_sigma[i] = _sigma(list_pd[i])
        if name_save:
            np.savetxt(name_sigma, vec_sigma, delimiter='\t')
    return np.median(vec_sigma)


def make_mat_distance(mat_linear):
    # distance is squared
    num_pd = mat_linear.shape[0]
    mat = np.zeros((num_pd, num_pd))
    for i in range(num_pd):
        for j in range(i):
            mat[i, j] = (
                mat_linear[i, i] + mat_linear[j, j] - 2 * mat_linear[i, j])
            mat[j, i] = mat[i, j]

    vec = np.zeros(int(num_pd * (num_pd - 1) / 2))
    idx_temp = 0
    for i in range(num_pd):
        for j in range(i):
            vec[idx_temp] = mat[i, j]
            idx_temp += 1
    return mat, np.median(vec)


def make_mat_gram(mat_linear, name_rkhs="Gaussian", normalize=True):
    if normalize:
        val_median = np.median(mat_linear)
        if (val_median > 1e+3) or (val_median < 1e-3):
            mat_linear /= val_median
        else:
            pass
    else:
        pass

    mat_distance, val_tau = make_mat_distance(mat_linear)
    if name_rkhs == "Gaussian":
        num_pd = len(mat_distance)
        mat_gram = np.ones((num_pd, num_pd))
        for i in range(num_pd):
            for j in range(i):
                mat_gram[i, j] = np.exp(
                    -1.0 * mat_distance[i, j] / (2.0 * val_tau))
                mat_gram[j, i] = mat_gram[i, j]
    else:
        mat_gram = mat_linear
    return mat_gram, mat_distance, val_tau


def plot_gram(mat_gram, name_save=None, show=False):
    plt.figure()
    plt.imshow(mat_gram, interpolation="nearest", cmap="YlOrRd")

    if name_save is not None:
        plt.savefig(name_save)
    else:
        pass

    if show:
        plt.show()
    else:
        pass

    plt.close()
