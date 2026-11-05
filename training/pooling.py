"""Static/exportable and shared pooling use identical feature-vector ordering."""
import tensorflow as tf


def static_pool(inputs, mode):
    if mode == 'avg':
        return tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(inputs)
    if mode == 'max':
        return tf.keras.layers.GlobalMaxPooling2D(name='global_max_pooling')(inputs)
    avg = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(inputs)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name='global_max_pooling')(inputs)
    return tf.keras.layers.Concatenate(name='mixed_global_pooling')([avg, maximum])


def shared_pool(inputs, mode):
    if mode == 'avg':
        return tf.reduce_mean(inputs, axis=[1, 2])
    if mode == 'max':
        return tf.reduce_max(inputs, axis=[1, 2])
    return tf.concat([tf.reduce_mean(inputs, axis=[1, 2]), tf.reduce_max(inputs, axis=[1, 2])], axis=-1)


def classifier_prefix(kernel, count, maximum, mode):
    # mix layout is [all avg channels, all max channels]. The smaller classifier
    # needs TWO prefixes; taking the first 2*count rows would silently be wrong.
    if mode == 'mix':
        return tf.concat([kernel[:count], kernel[maximum:maximum + count]], axis=0)
    return kernel[:count]
