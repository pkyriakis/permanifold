from tqdm import tqdm
import numpy as np

# original kusano
from baselines.kusano import tda


def make_rff(list_pd, list_weight, val_sigma, num_rff, tqdm_bar):
    num_pd = len(list_pd)
    mat_rff = np.empty((num_pd, num_rff))
    mat_cov = np.array([[np.power(val_sigma, -2), 0],
                        [0, np.power(val_sigma, -2)]])

    mat_z = np.random.multivariate_normal([0, 0], mat_cov, num_rff)
    vec_b = np.random.uniform(0, 2 * np.pi, num_rff)
    if tqdm_bar:
        print("==== Random Fourier features ====")
        process_bar = tqdm(total=int(num_pd))
        for k in range(num_pd):
            process_bar.set_description("%s" % k)
            mat_rff[k, :] = np.dot(
                list_weight[k], np.cos(np.inner(list_pd[k], mat_z) + vec_b))
            process_bar.update(1)
        process_bar.close()
    else:
        for k in range(num_pd):
            mat_rff[k, :] = np.dot(
                list_weight[k], np.cos(np.inner(list_pd[k], mat_z) + vec_b))
    return mat_rff


class Kernel:
    def __init__(self, list_pd, func_kernel, func_weight, val_sigma,
                 name_rkhs="Gaussian", approx=True, num_rff=int(1e+4),
                 tqdm_bar=False):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.func_kernel = func_kernel
        self.func_weight = func_weight
        self.val_sigma = val_sigma
        self.name_rkhs = name_rkhs
        self.approx = approx
        self.num_rff = num_rff
        self.tqdm_bar = tqdm_bar

        self.val_tau = None
        self.mat_distance = None

    def __vector_weight(self, mat_pd):
        num_point = mat_pd.shape[0]
        vec_weight = np.empty(num_point)
        for i in range(num_point):
            vec_weight[i] = self.func_weight(mat_pd[i, :])
        return vec_weight

    def __list_vector_weight(self):
        list_weight = []
        for i in range(self.num_pd):
            list_weight.append(self.__vector_weight(self.__list_pd[i]))
        return list_weight

    def __kernel_linear(self, mat_pd_1, mat_pd_2, vec_weight_1, vec_weight_2):
        val_sum = 0.0
        num_point_1 = mat_pd_1.shape[0]
        num_point_2 = mat_pd_2.shape[0]
        for i in range(num_point_1):
            for j in range(num_point_2):
                val_sum += (vec_weight_1[i] * vec_weight_2[j]
                            * self.func_kernel(mat_pd_1[i, :], mat_pd_2[j, :]))
        return val_sum

    def __matrix_linear(self):
        __mat_linear = np.empty((self.num_pd, self.num_pd))
        list_weight = self.__list_vector_weight()
        if self.tqdm_bar:
            print("==== gram matrix of weighted kernel ====")
            process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
            for i in range(self.num_pd):
                for j in range(i + 1):
                    process_bar.set_description("(%s, %s)" % (i, j))
                    __mat_linear[i, j] = self.__kernel_linear(
                        self.__list_pd[i], self.__list_pd[j],
                        list_weight[i], list_weight[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
                    process_bar.update(1)
            process_bar.close()
        else:
            for i in range(self.num_pd):
                for j in range(i + 1):
                    __mat_linear[i, j] = self.__kernel_linear(
                        self.__list_pd[i], self.__list_pd[j],
                        list_weight[i], list_weight[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
        return __mat_linear

    def gram(self):
        if self.approx:
            if self.val_sigma is None:
                print("approx=True is allowed only if func_kernel=Gaussian")
                mat_linear = np.diag(np.ones(self.num_pd))
            else:
                mat_rff = make_rff(self.__list_pd, self.__list_vector_weight(),
                                   self.val_sigma, self.num_rff, self.tqdm_bar)
                mat_linear = np.inner(mat_rff, mat_rff)
        else:
            mat_linear = self.__matrix_linear()

        mat_gram, self.mat_distance, self.val_tau = tda.make_mat_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def kernel(self, mat_pd_1, mat_pd_2, name_rkhs=None):
        vec_weight_1 = self.__vector_weight(mat_pd_1)
        vec_weight_2 = self.__vector_weight(mat_pd_2)
        val_c = self.__kernel_linear(
            mat_pd_1, mat_pd_2, vec_weight_1, vec_weight_2)

        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        if name_rkhs == "Gaussian":
            val_a = self.__kernel_linear(
                mat_pd_1, mat_pd_1, vec_weight_1, vec_weight_1)
            val_b = self.__kernel_linear(
                mat_pd_2, mat_pd_2, vec_weight_2, vec_weight_2)
            return np.exp(-(val_a + val_b - 2.0 * val_c) / (2.0 * self.val_tau))
        else:
            return val_c


def transpose(mat_pd):
    num_point = mat_pd.shape[0]
    mat_pd_trans = np.empty((num_point, 2))
    for i in range(num_point):
        mat_pd_trans[i, 0] = mat_pd[i, 1]
        mat_pd_trans[i, 1] = mat_pd[i, 0]
    return mat_pd_trans


class KernelPss:
    def __init__(self, list_pd, val_sigma, name_rkhs="Linear", approx=True,
                 num_rff=int(1e+4), tqdm_bar=False):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.val_sigma = 2 * val_sigma  # temporary
        self.name_rkhs = name_rkhs
        self.approx = approx
        self.num_rff = num_rff
        self.tqdm_bar = tqdm_bar

        self.val_tau = None
        self.mat_distance = None
        self.__list_pd_tilde = self.__list_tilde()

    def __func_gauss(self, vec_bd_1, vec_bd_2):
        val_dist = np.power(np.linalg.norm(vec_bd_1 - vec_bd_2, 2), 2)
        return np.exp(-1.0 * val_dist / (2.0 * np.power(self.val_sigma, 2)))

    def __list_tilde(self):
        list_tilde = []
        for i in range(self.num_pd):
            list_tilde.append(
                np.r_[self.__list_pd[i], transpose(self.__list_pd[i])])
        return list_tilde

    def __kernel_linear(self, mat_pd_1, mat_pd_2):
        val_sum = 0.0
        num_point_1 = mat_pd_1.shape[0]
        num_point_2 = mat_pd_2.shape[0]
        mat_pd_2_trans = transpose(mat_pd_2)
        for i in range(num_point_1):
            for j in range(num_point_2):
                val_sum += (self.__func_gauss(mat_pd_1[i, :], mat_pd_2[j, :])
                            - self.__func_gauss(mat_pd_1[i, :],
                                                mat_pd_2_trans[j, :]))
        return val_sum

    def __matrix_linear(self):
        __mat_linear = np.empty((self.num_pd, self.num_pd))
        if self.tqdm_bar:
            print("==== gram matrix of pss kernel ====")
            process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
            for i in range(self.num_pd):
                for j in range(i + 1):
                    process_bar.set_description("(%s, %s)" % (i, j))
                    __mat_linear[i, j] = self.__kernel_linear(
                        self.__list_pd[i], self.__list_pd[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
                    process_bar.update(1)
            process_bar.close()
        else:
            for i in range(self.num_pd):
                for j in range(i + 1):
                    __mat_linear[i, j] = self.__kernel_linear(
                        self.__list_pd[i], self.__list_pd[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
        return __mat_linear

    def __list_vector_weight(self):
        __list_weight = []
        for i in range(self.num_pd):
            num_point = self.__list_pd[i].shape[0]
            __list_weight.append(
                np.r_[np.ones(num_point), -1 * np.ones(num_point)])
        return __list_weight

    def gram(self):
        if self.approx:
            mat_rff = make_rff(
                self.__list_pd_tilde, self.__list_vector_weight(),
                self.val_sigma, self.num_rff, self.tqdm_bar)
            mat_linear = np.inner(mat_rff, mat_rff)
        else:
            mat_linear = self.__matrix_linear()

        if self.name_rkhs == "Universal":
            mat_linear = tda.make_mat_gram(mat_linear, "Linear")[0]  # normalize
            mat_gram = np.empty((self.num_pd, self.num_pd))
            for i in range(self.num_pd):
                for j in range(self.num_pd):
                    mat_gram[i, j] = np.exp(mat_linear[i, j])
        else:
            mat_gram, self.mat_distance, self.val_tau = tda.make_mat_gram(
                mat_linear, self.name_rkhs)
        return mat_gram

    def kernel(self, mat_pd_1, mat_pd_2, name_rkhs=None):
        val_c = self.__kernel_linear(mat_pd_1, mat_pd_2)

        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        if name_rkhs == "Gaussian":
            val_a = self.__kernel_linear(mat_pd_1, mat_pd_1)
            val_b = self.__kernel_linear(mat_pd_2, mat_pd_2)
            return np.exp(-(val_a + val_b - 2.0 * val_c) / (2.0 * self.val_tau))
        elif name_rkhs == "Universal":
            return np.exp(val_c)
        else:
            return val_c
