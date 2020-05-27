import math
import tensorflow as tf


class Poincare:
    '''
        Defines several helper kusano needed by the Poincare model
        such as exp and log maps and the coordinate chart and parameterization

        Note: Some methods were adapted from @https://github.com/dalab/hyperbolic_nn
    '''

    def __init__(self, max_num_of_points, man_dim, K):
        self.PROJ_EPS = 1e-5
        self.EPS = 1e-5
        self.MAX_TANH_ARG = 15.0
        self.man_dim = man_dim
        self.max_num_of_points = max_num_of_points
        self.K = K

    def dot(self, x, y):
        return tf.reduce_sum(x * y, axis=-1, keepdims=True)

    def norm(self, x):
        return tf.norm(x, axis=-1, keepdims=True)

    def lambda_x(self, x, c):
        return 2. / (1 - c * self.dot(x, x))

    def atanh(self, x):
        return tf.atanh(tf.minimum(x, 1. - self.EPS))  # Only works for positive real x.

    def tanh(self, x):
        return tf.tanh(tf.minimum(tf.maximum(x, -self.MAX_TANH_ARG), self.MAX_TANH_ARG))

    def mob_add(self, u, v, c):
        v = v + self.EPS
        tf_dot_u_v = 2. * c * self.dot(u, v)
        tf_norm_u_sq = c * self.dot(u, u)
        tf_norm_v_sq = c * self.dot(v, v)
        denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
        result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
        return self.project_to_manifold(result, c)

    def project_to_manifold(self, x, c=1.):
        # Projection op. Need to x is inside the unit ball.
        return tf.clip_by_norm(t=x, clip_norm=(1. - self.PROJ_EPS) / tf.sqrt(c), axes=[0])

    def exp_map_x(self, x, v, c=1.):
        v = v + self.EPS
        norm_v = self.norm(v)
        second_term = (self.tanh(tf.sqrt(c) * self.lambda_x(x, c) * norm_v / 2) / (tf.sqrt(c) * norm_v)) * v
        return self.mob_add(x, second_term, c)

    def log_map_x(self, x, y, c=1.):
        diff = self.mob_add(-x, y, c) + self.EPS
        norm_diff = self.norm(diff)
        lam = self.lambda_x(x, c)
        val = (((2. / tf.sqrt(c)) / lam) * self.atanh(tf.sqrt(c) * norm_diff) / norm_diff) * diff
        return val

    def parametrization(self, y):
        '''
            Transforms the Euclidean point y onto the manifold
            m is the man_dim
        '''
        m = self.man_dim
        sliced_y = tf.split(y, num_or_size_splits=m, axis=-1)
        x = tf.norm(y, axis=-1, keepdims=True)
        for i in tf.range(1, m, dtype=tf.int32):
            tf.autograph.experimental.set_loop_options(shape_invariants=
                                                       [(x, tf.TensorShape([None, self.max_num_of_points, None]))])
            ind = tf.range(i - 1, m)
            gathered = tf.gather(y, indices=ind, axis=-1)
            gathered = tf.reshape(gathered, shape=[-1, self.max_num_of_points, m - i + 1])
            den = tf.norm(gathered, axis=-1, keepdims=True) + self.EPS

            gathered = tf.gather(sliced_y, indices=i - 1)
            nom = tf.reshape(gathered, shape=[-1, self.max_num_of_points, 1])
            x_i = tf.acos(nom / den)
            if i == m - 1:
                x_i = tf.where(x_i > 0, x_i, 2 * math.pi - x_i)
            x = tf.concat([x, x_i], axis=-1)
        return tf.where(tf.math.is_nan(x), 0., x)

    def chart(self, x):
        '''
            Transforms the manifold point x to the Euclidean space
            m is the man_dim
        '''
        m = self.man_dim
        x_0 = tf.gather(x, axis=-1, indices=0)
        x_1 = tf.gather(x, axis=-1, indices=1)
        y = tf.multiply(x_0, tf.cos(x_1))
        y = tf.expand_dims(y, axis=-1)
        for i in tf.range(1, m, dtype=tf.int32):
            tf.autograph.experimental.set_loop_options(shape_invariants=
                                                       [(y, tf.TensorShape([None, self.K, None]))])
            y_i = tf.gather(x, axis=-1, indices=0)
            if i == m - 1:
                for j in tf.range(1, m):
                    y_i = tf.multiply(y_i, tf.sin(tf.gather(x, axis=-1, indices=j)))
            else:
                for j in tf.range(1, i + 1):
                    y_i = tf.multiply(y_i, tf.sin(tf.gather(x, axis=-1, indices=j)))
                y_i = tf.multiply(y_i, tf.cos(tf.gather(x, axis=-1, indices=i + 1)))
            y_i = tf.expand_dims(y_i, axis=-1)
            y = tf.concat([y, y_i], axis=-1)
        return y


class Euclidean:
    '''
            Defines helper kusano needed by the Euclidean model
    '''

    def project_to_manifold(self, x):
        return x

    def exp_map_x(self, x, v):
        return tf.add(x, v)

    def log_map_x(self, x, y):
        return tf.add(y, -x)

    def chart(self, x):
        return x

    def parametrization(self, y):
        return y
