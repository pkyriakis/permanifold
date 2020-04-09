import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import models


# x = tf.Variable([[1.1,0.0],[0.0,1.0]])
#
# with tf.GradientTape(persistent=True) as t:
#   t.watch(x)
#   z = tf.multiply(x, x) + tf.ones((2,2)) +3 + tf.norm(x)
#   print(z.numpy())

dgm = [[1.1,1.2, 1.3],[1.0, 0.0,1.0]]
pm = models.PManifoldLayer(3,2)
a = pm(dgm)
print(a)
