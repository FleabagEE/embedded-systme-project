"""Checkpoint, learning-rate and early-stop control share one explicit monitor."""
import tensorflow as tf


def training_callbacks(model, optimizer, settings, checkpoint):
    monitor = settings.get('monitor', 'val_loss')
    mode = 'min' if monitor == 'val_loss' else 'max'
    model.optimizer = optimizer
    model.stop_training = False
    early = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                             patience=settings['early_stopping_patience'])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor=monitor, mode=mode,
                                                    save_best_only=True, save_weights_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, min_delta=0,
                                                      patience=settings.get('lr_patience', 3),
                                                      factor=settings.get('lr_factor', 0.5),
                                                      min_lr=settings.get('min_learning_rate', 1e-5)), early]
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()
    return callbacks, early


def epoch_message(epoch, epochs, row, settings, durations, widths):
    duration_scores = ' '.join(f"{d:g}s={row[f'val_accuracy_{d:g}s']:.2%}" for d in durations)
    width_scores = (' widths[' + ' '.join(f"{w}={row[f'val_accuracy_{w}']:.2%}" for w in widths) + ']') if len(widths) > 1 else ''
    return (f"Epoch {epoch}/{epochs}: loss={row['total_loss']:.5f} acc={row['accuracy']:.2%} "
            f"val_loss={row['val_loss']:.5f} val_acc={row['val_accuracy']:.2%} [{duration_scores}]"
            f"{width_scores} patience={row['patience_used']}/{settings['early_stopping_patience']} "
            f"remaining={row['patience_remaining']} best_epoch={row['best_epoch']} "
            f"lr={row['learning_rate']:.3g}")
