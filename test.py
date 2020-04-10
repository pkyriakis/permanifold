import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import layers
import numpy as np
import utils


# x = tf.Variable([[1.1,0.0],[0.0,1.0]])
#
# with tf.GradientTape(persistent=True) as t:
#   t.watch(x)
#   z = tf.multiply(x, x) + tf.ones((2,2)) +3 + tf.norm(x)
#   print(z.numpy())

dgm = np.array([[0, 1.1, 1.2], [1, 2.0, 3.2]], dtype=np.float32)
pm = layers.PManifoldLayer(3, 2, 2)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)

print(pm(dgm))


# y = tf.Variable([1.,2., 3.0])
# theta = tf.Variable([3.,2.,0.5],trainable=True)
# with tf.GradientTape() as t:
#     t.watch(theta)
#     x = utils.Poincare.tf_parametrization(y,theta)
#     print(x)
#     #x =tf.multiply(theta[0], tf.norm(y))
# print(t.gradient(x,theta))
#
# steps_acc = tf.zeros([5,], dtype=tf.dtypes.float32)
# times_acc = tf.zeros((4,), dtype=tf.dtypes.float32)
# print(steps_acc)
# print(times_acc)
# cc = tf.concat([steps_acc,times_acc], axis=0)
# print(cc)