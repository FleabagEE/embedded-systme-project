"""Accuracy-only checkpoint control and reusable epoch-level progress reporting."""
from collections import deque
import time
import tensorflow as tf
from training.utils import WIDTHS, duration_token


def print_progress(message):
    # Subprocess stdout may be piped by Colab; do not defer epochs until export.
    print(message, flush=True)


class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, early, epochs, settings, durations, widths, clock=time.perf_counter, printer=print_progress):
        super().__init__()
        self.early, self.epochs, self.settings = early, epochs, settings
        self.durations, self.widths = durations, widths
        self.clock, self.printer = clock, printer
        self.recent = deque(maxlen=settings.get('eta_window',5))
        self.min_epochs = settings.get('eta_min_epochs',2)
        self.elapsed = []

    def on_train_begin(self, logs=None):
        self.started = self.clock()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started = self.clock()

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        now = self.clock(); seconds = max(0.,now - self.epoch_started)
        self.elapsed.append(seconds); self.recent.append(seconds)
        remaining = 0 if self.model.stop_training else max(0,self.epochs - epoch - 1)
        eta = sum(self.recent)/len(self.recent)*remaining if len(self.elapsed) >= self.min_epochs else None
        used = int(self.early.wait)
        improved = int(self.early.best_epoch) == epoch
        logs.update(epoch_seconds=seconds, training_elapsed_seconds=max(0.,now-self.started), eta_seconds=eta,
                    best_val_accuracy=float(self.early.best), checkpoint_improved=improved,
                    patience_used=used, patience_remaining=max(0,self.settings['early_stopping_patience']-used),
                    best_epoch=int(self.early.best_epoch)+1, learning_rate=float(self.model.optimizer.learning_rate.numpy()))
        eta_text = 'pending (needs more epochs)' if eta is None else f'~{eta/60:.1f} min' if eta >= 60 else f'~{eta:.0f}s'
        self.printer(f"Epoch {epoch+1}/{self.epochs} | {seconds:.1f}s\n"
                     f"Train Acc: {logs['accuracy']:.2%} | Val Acc: {logs['val_accuracy']:.2%}\n"
                     f"Best Val: {logs['best_val_accuracy']:.2%} (Epoch {logs['best_epoch']})\n"
                     f"LR: {logs['learning_rate']:.5g} | Patience: {used}/{self.settings['early_stopping_patience']} "
                     f"(remaining {logs['patience_remaining']})\n"
                     f"ETA: {eta_text} | Checkpoint: {'Saved' if improved else 'No improvement'}")
        if len(self.widths) > 1 or len(self.durations) > 1:
            header = 'Validation Accuracy\n             ' + '  '.join(f'{d:.1f}s' for d in self.durations)
            lines = []
            for width in sorted(self.widths,key=lambda w:WIDTHS[w]):
                values = '  '.join(f"{logs[f'val_accuracy_{width}_{duration_token(d)}']:6.2%}" for d in self.durations)
                lines.append(f'Width {WIDTHS[width]:.2f}   {values}')
            self.printer(header + '\n' + '\n'.join(lines))


def training_callbacks(model, optimizer, settings, checkpoint, *, durations=None, widths=None, epochs=None):
    monitor, mode = 'val_accuracy', 'max'
    model.optimizer = optimizer
    model.stop_training = False
    # All switchable BN affine variables must be registered before a sandwich
    # step that may omit one width; later steps must not introduce new slots.
    optimizer.build(model.trainable_variables)
    early = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                             patience=settings['early_stopping_patience'])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor=monitor, mode=mode,
                                                    save_best_only=True, save_weights_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, min_delta=0,
                                                      patience=settings.get('lr_patience',3),
                                                      factor=settings.get('lr_factor',.5),
                                                      min_lr=settings.get('min_learning_rate',1e-5)), early]
    if durations is not None and widths is not None:
        callbacks.append(TrainingProgress(early,epochs or settings['epochs'],settings,durations,widths))
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()
    return callbacks, early
