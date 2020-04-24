import tensorflow as tf
import math

class Poincare:
    '''
        Defines several helper functions needed by the Poincare model
        such as exp and log maps and the coordinate chart and parameterization
    '''
    PROJ_EPS = 1e-5
    EPS = 1e-5
    MAX_TANH_ARG = 15.0

    def tf_dot(self, x, y):
        return tf.reduce_sum(x * y, keepdims=True)

    def tf_norm(self, x):
        return tf.norm(x, keepdims=True)


    def tf_lambda_x(self, x, c):
        return 2. / (1 - c * self.tf_dot(x,x))

    def tf_atanh(self, x):
        return tf.atanh(tf.minimum(x, 1. - self.EPS)) # Only works for positive real x.

    # Real x, not vector!
    def tf_tanh(self, x):
       return tf.tanh(tf.minimum(tf.maximum(x, -self.MAX_TANH_ARG), self.MAX_TANH_ARG))

    def tf_project_hyp_vecs(self, x, c):
        # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
        return tf.clip_by_norm(t=x, clip_norm=(1. - self.PROJ_EPS) / tf.sqrt(c), axes=[0])

    def tf_mob_add(self, u, v, c):
        v = v + self.EPS
        tf_dot_u_v = 2. * c * self.tf_dot(u, v)
        tf_norm_u_sq = c * self.tf_dot(u,u)
        tf_norm_v_sq = c * self.tf_dot(v,v)
        denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
        result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
        return self.tf_project_hyp_vecs(result, c)

    def tf_exp_map_x(self, x, v, c):
        v = v + self.EPS # Perturbe v to avoid dealing with v = 0
        norm_v = self.tf_norm(v)
        second_term = (self.tf_tanh(tf.sqrt(c) * self.tf_lambda_x(x, c) * norm_v / 2) / (tf.sqrt(c) * norm_v)) * v
        return self.tf_mob_add(x, second_term, c)

    def tf_log_map_x(self, x, y, c):
        diff = self.tf_mob_add(-x, y, c) + self.EPS
        norm_diff = self.tf_norm(diff)
        lam = self.tf_lambda_x(x, c)
        val = (((2. / tf.sqrt(c)) / lam) * self.tf_atanh(tf.sqrt(c) * norm_diff) / norm_diff) * diff
        return val

    def tf_parametrization(self, y, m):
        '''
            Transforms the Euclidean point y onto the manifold given params theta
        '''
        sliced_y = tf.split(y, num_or_size_splits=m, axis=-1)
        x = tf.norm(y, axis=-1, keepdims=True)
        for i in tf.range(1, m, dtype=tf.int32):
            tf.autograph.experimental.set_loop_options(shape_invariants=
                                                       [(x, tf.TensorShape([None,None,None]))])
            ind = tf.range(i - 1, m)
            gathered = tf.gather(y, indices=ind, axis=-1)
            den = tf.norm(gathered, axis=-1, keepdims=True) + self.EPS
            x_i = tf.acos(tf.gather(sliced_y,indices=i-1) / den)
            if i == m - 1:
                x_i = tf.where(x_i > 0, x_i, 2 * math.pi - x_i)
            x = tf.concat([x,x_i], axis=-1)
        return tf.where(tf.math.is_nan(x), 0., x)

    def tf_chart(self, x, m):
        '''
            Transforms the manifold point x to the Euclidean space
        '''
        sliced_x = tf.split(x, num_or_size_splits=m, axis=-1)
        y = tf.multiply(sliced_x[0], tf.cos(sliced_x[1]))
        for i in tf.range(1, m, dtype=tf.int32):
            tf.autograph.experimental.set_loop_options(shape_invariants=
                                                       [(y, tf.TensorShape([None, None, None]))])
            y_i = tf.gather(sliced_x, indices=0)
            if i == m - 1:
                for j in tf.range(1, m):
                    y_i = tf.multiply(y_i, tf.sin(tf.gather(sliced_x, indices=j)))
            else:
                for j in tf.range(1, i + 1):
                    y_i = tf.multiply(y_i, tf.sin(tf.gather(sliced_x, indices=j)))
                y_i = tf.multiply(y_i, tf.cos(tf.gather(sliced_x, indices=i+1)))
            y = tf.concat([y,y_i], axis=-1)
        return y

class Lorenz:
    '''
            Defines several helper functions needed by the Lorenz model
            such as exp and log maps and the coordinate chart and parameterization
    '''
    pass
