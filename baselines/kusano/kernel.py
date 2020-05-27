import numpy as np

# original kusano
from baselines.kusano import pwk
from kusano import image
from kusano import landscape
from kusano import sw


def function_weight(name_weight, val_c=1.0, val_p=1):
    if name_weight == "arctan":
        def func_weight(vec_bd):
            return np.arctan(np.power((vec_bd[1] - vec_bd[0]) / val_c, val_p))
    else:  # p-persistence
        def func_weight(vec_bd):
            return np.power(vec_bd[1] - vec_bd[0], val_p)
    return func_weight


def function_kernel(name_kernel, val_sigma=1.0):
    if (name_kernel == "Gaussian") and (val_sigma is not None):
        def func_kernel(vec_bd_1, vec_bd_2):
            val_dist = np.power(np.linalg.norm(vec_bd_1 - vec_bd_2, 2), 2)
            return np.exp(-1.0 * val_dist / (2.0 * np.power(val_sigma, 2)))
    else:  # linear kernel
        def func_kernel(vec_bd_1, vec_bd_2):
            return np.dot(vec_bd_1, vec_bd_2)
    return func_kernel


def plot_image(mat_pi, val_y=None, range_bd=None, diag=True, name_save=None,
               show=True):
    return image.plot(mat_pi, val_y, range_bd, diag, name_save, show)


def plot_landscape(list_pd, int_k=1, num_slice=100, range_bd=None, val_y=None,
                   show=True, name_save=None):
    return landscape.plot_average(list_pd, int_k, num_slice, range_bd, val_y,
                                  show, name_save)


def gram(list_pd, name_gram="pwk", name_rkhs="Gaussian", name_kernel="Gaussian",
         val_sigma=None, name_weight="arctan", val_c=1.0, val_p=1,
         approx=True, num_mesh=80, tqdm_bar=False, num_sw=36, num_rff=int(1e+4),
         range_bd=None):

    func_kernel = function_kernel(name_kernel, val_sigma)
    func_weight = function_weight(name_weight, val_c, val_p)

    if (name_gram == "pwk") and (val_sigma is not None):
        mat_gram = pwk.Kernel(
            list_pd, func_kernel, func_weight, val_sigma, name_rkhs, approx,
            num_rff, tqdm_bar).gram()

    elif (name_gram == "pssk") and (val_sigma is not None):
        mat_gram = pwk.KernelPss(
            list_pd, val_sigma, name_rkhs, approx, num_rff, tqdm_bar).gram()

    elif (name_gram == "u-pssk") and (val_sigma is not None):
        mat_gram = pwk.KernelPss(
            list_pd, val_sigma, "Universal", approx, num_rff, tqdm_bar).gram()

    elif (name_gram == "image") and (val_sigma is not None):
        mat_gram = image.Kernel(
            list_pd, func_weight, val_sigma, num_mesh, name_rkhs, range_bd,
            tqdm_bar).gram()

    elif name_gram == "landscape":
        mat_gram = landscape.Kernel(
            list_pd, name_rkhs, range_bd, tqdm_bar=True).gram()

    elif name_gram == "sw":
        mat_gram = sw.Kernel(list_pd, num_sw, tqdm_bar).gram()

    else:
        print("something wrong")
        print(name_gram, val_sigma, name_gram)
        mat_gram = np.diag(np.ones(len(list_pd)))

    if name_gram == "pwk":
        name_mat = "%s_%s_%s_%s" % (
            name_gram, name_kernel, name_weight, name_rkhs)
    elif name_gram == "pssk" or name_gram == "u-pssk":
        name_mat = "%s_%s_%s" % (name_gram, name_kernel, name_rkhs)
    elif name_gram == "image":
        name_mat = "%s_%s_%s" % (name_gram, name_weight, name_rkhs)
    else:  # landscape and sw
        name_mat = "%s_%s" % (name_gram, name_rkhs)

    return mat_gram, name_mat
