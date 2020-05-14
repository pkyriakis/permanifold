import math
import tensorflow as tf
import json


class Poincare:
    '''
        Defines several helper functions needed by the Poincare model
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

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

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


class Lorenz:
    '''
                Defines several helper functions needed by the Lorenz model
                such as exp and log maps and the coordinate chart and parameterization
    '''

    def __init__(self, max_num_of_points, man_dim, K):
        self.EPS = 1e-5
        self.poincare = Poincare(max_num_of_points, man_dim, K)
        self.man_dim = man_dim
        self.max_num_of_points = max_num_of_points
        self.K = K

    def cosh(self, x):
        '''
            Cosh can cause the optimization to be very slow cuz it has high valued gradient
            We use first order Taylor approximation
        '''
        apprx = 1 + tf.math.square(x)/2.
        return apprx

    def sinh(self, x):
        '''
            Same as above
        '''
        appr = 1 - tf.math.pow(x, 3) / 6.
        return appr

    def dot_g(self, x, y):
        '''
            Returns dot product under Lorenzian metric tensor
        '''
        tmp1 = -tf.gather(y, indices=0, axis=-1)
        tmp1 = tf.expand_dims(tmp1, axis=-1)
        tmp2 = tf.gather(y, indices=tf.range(1, self.man_dim), axis=-1)
        y = tf.concat([tmp1, tmp2], axis=-1)
        ret = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        return ret

    def project_to_manifold(self, x):
        # Projection op. Need to make sure x is on the hyperbolic sheet
        # We use the Poincare cuz it's simpler to project a point there

        # Transfer to Poincare
        den = tf.expand_dims(tf.gather(x, indices=0, axis=-1) + 1, axis=-1) + 1 + self.EPS
        poinc_x = x / den

        # Project
        proj_poinc_x = self.poincare.project_to_manifold(poinc_x)

        # Transfer back to Lorenz
        den = tf.expand_dims(1 - tf.norm(proj_poinc_x, axis=-1), axis=-1) + self.EPS
        nom = 1 + tf.square(tf.norm(proj_poinc_x, axis=-1, keepdims=True))
        rest = 2 * tf.gather(proj_poinc_x, indices=tf.range(1, self.man_dim), axis=-1)
        nom = tf.concat([nom, rest], axis=-1)
        return nom / den

    def exp_map_x(self, x, v):
        dt = self.dot_g(v, v)
        out1 = self.cosh(dt) * x
        out2 = self.sinh(dt) * x / (dt + self.EPS)
        return tf.add(out1, out2)

    def log_map_x(self, x, y):
        arg = tf.clip_by_value(-self.dot_g(x, y), 1., tf.float32.max)
        nom = tf.acosh(arg)
        sq = tf.clip_by_value(self.dot_g(x, y) - 1, 0., tf.float32.max)
        den = tf.sqrt(sq) + self.EPS
        fact = y + x*self.dot_g(x, y)
        ret = fact * nom / den
        return fact

    def parametrization(self, y):
        '''
            Transforms the Euclidean point y onto the manifold
        '''
        m = self.man_dim
        r_sq = self.dot_g(y, y)
        clipped_r = tf.clip_by_value(r_sq, 0., tf.float32.max)
        x = tf.sqrt(clipped_r)

        for i in tf.range(1, m, dtype=tf.int32):
            tf.autograph.experimental.set_loop_options(shape_invariants=
                                                       [(x, tf.TensorShape([None, self.max_num_of_points, None]))])
            ind = tf.range(i - 1, m)
            gathered = tf.gather(y, indices=ind, axis=-1)
            gathered = tf.reshape(gathered, shape=[-1, self.max_num_of_points, m - i + 1])
            den = tf.norm(gathered, axis=-1, keepdims=True) + self.EPS

            nom = tf.gather(y, axis=-1, indices=i - 1)
            nom = tf.expand_dims(nom, axis=-1)
            if i == 1:
                clipped = tf.clip_by_value( nom / den, 1., tf.float32.max)
                x_i = tf.acosh(clipped)
            else:
                x_i = tf.acos(nom / den)
            if i == m - 1:
                x_i = tf.where(x_i > 0., x_i, 2 * math.pi - x_i)
            x = tf.concat([x, x_i], axis=-1)
        return tf.where(tf.math.is_nan(x), 0., x)

    def chart(self, x):
        '''
            Transforms the manifold point x to the Euclidean space
        '''
        m = self.man_dim
        x_0 = tf.gather(x, axis=-1, indices=0)
        x_1 = tf.gather(x, axis=-1, indices=1)
        y = tf.multiply(x_0, self.cosh(x_1))
        y = tf.expand_dims(y, axis=-1)
        for i in tf.range(1, m, dtype=tf.int32):
            tf.autograph.experimental.set_loop_options(shape_invariants=
                                                       [(y, tf.TensorShape([None, self.K, None]))])
            y_i = tf.gather(x, axis=-1, indices=0)
            y_i = tf.multiply(y_i, self.sinh(tf.gather(x, axis=-1, indices=1)))
            if i == m - 1:
                for j in tf.range(2, m):
                    y_i = tf.multiply(y_i, tf.sin(tf.gather(x, axis=-1, indices=j)))
            else:
                for j in tf.range(2, i + 1):
                    y_i = tf.multiply(y_i, tf.sin(tf.gather(x, axis=-1, indices=j)))
                y_i = tf.multiply(y_i, tf.cos(tf.gather(x, axis=-1, indices=i + 1)))
            y_i = tf.expand_dims(y_i, axis=-1)
            y = tf.concat([y, y_i], axis=-1)
        return y


class Euclidean:
    '''
            Defines helper functions needed by the Euclidean model
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
