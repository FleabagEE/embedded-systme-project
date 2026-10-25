"""One GradientTape and one optimizer update for all shared-width forwards."""
import tensorflow as tf


def width_losses(model, x, labels, weights=(1., 1., 1.), training=True):
    losses, logits = {}, {}
    for width in ('1', '5', '25'):
        logits[width] = model(x, width=width, training=training)
        losses[width] = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits[width], from_logits=True))
    total = tf.add_n([losses[w] * weight for w, weight in zip(('1', '5', '25'), weights)])
    return total, losses, logits


def shared_train_step(model, optimizer, x, labels, weights=(1., 1., 1.)):
    with tf.GradientTape() as tape:
        total, losses, logits = width_losses(model, x, labels, weights)
    gradients = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total, losses, logits
