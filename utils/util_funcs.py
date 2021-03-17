import tensorflow as tf


# TODO - unit test!
# Tested manually on different cases seems to work
def reduce_mean_masked(x, mask, axis):
    x = x * tf.cast(mask, dtype=tf.float32)
    m = tf.reduce_sum(x, axis=axis) / tf.cast(tf.reduce_sum(mask, axis=axis), tf.float32)
    return m


def reduce_sum_masked(x, mask, axis):
    x = x * tf.cast(mask, dtype=tf.float32)
    m = tf.reduce_sum(x, axis=axis)
    return m
