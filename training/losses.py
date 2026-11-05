"""One GradientTape and one optimizer update for all shared-width forwards."""
import tensorflow as tf


def width_losses(model, x, labels, weights=(1., 1., 1.), training=True, widths=None):
    losses, logits = {}, {}
    widths = list(('1', '5', '25') if widths is None else widths)
    weight_map = weights if isinstance(weights, dict) else dict(zip(('1','5','25'), weights))
    for width in widths:
        logits[width] = model(x, width=width, training=training)
        losses[width] = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits[width], from_logits=getattr(model,'output_logits',True)))
    total = tf.add_n([losses[w] * weight_map[w] for w in widths])
    return total, losses, logits


def shared_train_step(model, optimizer, x, labels, weights=(1., 1., 1.), widths=None):
    with tf.GradientTape() as tape:
        total, losses, logits = width_losses(model, x, labels, weights, widths=widths)
    gradients = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients((g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None)
    return total, losses, logits


def sample_widths(config, rng):
    from training.utils import selected_widths, WIDTHS
    widths = selected_widths(config)
    if config['all_width_training'].get('strategy','all') == 'all' or len(widths) < 3:
        return widths
    ordered = sorted(widths,key=lambda w:WIDTHS[w],reverse=True)
    middle = ordered[1:-1]
    count = min(config['all_width_training'].get('sandwich_random_widths',1),len(middle))
    return [ordered[0],ordered[-1],*list(rng.choice(middle,size=count,replace=False))]
