"""
Helpers for the neural network notebooks.

TensorFlow is imported inside the functions that need it, so that notebooks
1 to 4 run even on a machine where TensorFlow is not installed.

Shape conventions (these cause most beginner errors, so learn them now):

    Tabular model (Ridge, gradient boosting, MLP)  X has shape (samples, features)
    Sequence model (LSTM, GRU, 1D CNN, Transformer) X has shape (samples, timesteps, features)

For a sequence model each sample is a small "movie" of the last `timesteps`
days, and each frame of the movie has `features` numbers.
"""

import os
import random

import numpy as np

from . import config


def set_seed(seed=None):
    """Make runs repeatable. (Exact repeatability across different computers is not guaranteed.)"""
    seed = config.SEED if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import keras

    keras.utils.set_random_seed(seed)


def make_windows(features, target, window):
    """
    Turn a table into overlapping windows.

    features  array of shape (n_days, n_features), already scaled
    target    array of shape (n_days,)
    window    how many past days go into one sample

    Sample k contains days k, k+1, ..., k+window-1 and its label is the target
    of the LAST day in the window, so no label ever comes from a day earlier
    than the information it is based on.

    Returns X of shape (n_days - window + 1, window, n_features) and y of
    shape (n_days - window + 1,). The first `window - 1` days have no complete
    window and are lost.
    """
    features = np.asarray(features)
    target = np.asarray(target)
    n_days, n_features = features.shape
    n_samples = n_days - window + 1
    X = np.empty((n_samples, window, n_features), dtype="float32")
    for k in range(n_samples):
        X[k] = features[k : k + window]
    y = target[window - 1 :].astype("float32")
    return X, y


# ---------------------------------------------------------------------------
# Model builders. Every builder returns a compiled Keras model that takes a
# window of shape (timesteps, features) and outputs ONE number: the predicted
# (standardised) return for the next day.
# ---------------------------------------------------------------------------
def _compile(model, learning_rate):
    import keras

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_mlp(timesteps, n_features, units=(32, 16), dropout=0.2, l2=1e-4, learning_rate=1e-3):
    """
    Feed-forward network (FNN / MLP). It flattens the window into one long
    row of numbers and has no idea that those numbers form a sequence.
    """
    import keras
    from keras import layers, regularizers

    layer_list = [layers.Input(shape=(timesteps, n_features)), layers.Flatten()]
    for width in units:
        layer_list.append(layers.Dense(width, activation="relu", kernel_regularizer=regularizers.l2(l2)))
        layer_list.append(layers.Dropout(dropout))
    layer_list.append(layers.Dense(1))
    return _compile(keras.Sequential(layer_list), learning_rate)


def build_lstm(timesteps, n_features, units=32, dropout=0.2, l2=1e-4, learning_rate=1e-3):
    """
    Long Short-Term Memory network. Reads the window one day at a time and
    carries a memory (the cell state) from day to day. Only the final state
    is used to make the prediction.
    """
    import keras
    from keras import layers, regularizers

    return _compile(
        keras.Sequential(
            [
                layers.Input(shape=(timesteps, n_features)),
                layers.LSTM(units, kernel_regularizer=regularizers.l2(l2)),
                layers.Dropout(dropout),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        ),
        learning_rate,
    )


def build_gru(timesteps, n_features, units=32, dropout=0.2, l2=1e-4, learning_rate=1e-3):
    """Gated Recurrent Unit: a lighter cousin of the LSTM with fewer gates and fewer weights."""
    import keras
    from keras import layers, regularizers

    return _compile(
        keras.Sequential(
            [
                layers.Input(shape=(timesteps, n_features)),
                layers.GRU(units, kernel_regularizer=regularizers.l2(l2)),
                layers.Dropout(dropout),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        ),
        learning_rate,
    )


def build_cnn(timesteps, n_features, filters=32, kernel_size=3, dropout=0.2, l2=1e-4, learning_rate=1e-3):
    """
    1D convolutional network. A small filter slides along the time axis and
    learns short local patterns (for example "three days of falling volume
    followed by a rise"). Global average pooling then summarises the window.
    """
    import keras
    from keras import layers, regularizers

    return _compile(
        keras.Sequential(
            [
                layers.Input(shape=(timesteps, n_features)),
                layers.Conv1D(filters, kernel_size, activation="relu", padding="causal",
                              kernel_regularizer=regularizers.l2(l2)),
                layers.Conv1D(filters, kernel_size, activation="relu", padding="causal",
                              kernel_regularizer=regularizers.l2(l2)),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(dropout),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        ),
        learning_rate,
    )


def build_transformer(timesteps, n_features, d_model=32, heads=2, ff_units=64, dropout=0.2, learning_rate=1e-3):
    """
    A very small Transformer encoder.

    1. A Dense layer projects each day's features to d_model numbers.
    2. A learned position embedding tells the model which day is which
       (attention by itself has no sense of order).
    3. Multi-head self-attention lets every day look at every other day in
       the window and decide how much to borrow from it.
    4. A feed-forward block and residual connections, then average pooling.
    """
    import keras
    from keras import layers

    class AddPositionEmbedding(layers.Layer):
        """Adds a learned vector for each position (day) in the window."""

        def __init__(self, length, width, **kwargs):
            super().__init__(**kwargs)
            self.length, self.width = length, width
            self.embedding = layers.Embedding(input_dim=length, output_dim=width)

        def call(self, x):
            positions = keras.ops.arange(self.length)
            return x + self.embedding(positions)

    inputs = layers.Input(shape=(timesteps, n_features))
    x = layers.Dense(d_model)(inputs)
    x = AddPositionEmbedding(timesteps, d_model)(x)

    attention = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attention)
    feed_forward = layers.Dense(ff_units, activation="relu")(x)
    feed_forward = layers.Dense(d_model)(feed_forward)
    x = layers.LayerNormalization()(x + layers.Dropout(dropout)(feed_forward))

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    return _compile(keras.Model(inputs, outputs), learning_rate)


BUILDERS = {
    "MLP": build_mlp,
    "LSTM": build_lstm,
    "GRU": build_gru,
    "CNN": build_cnn,
    "Transformer": build_transformer,
}


def fit_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, patience=10, verbose=0):
    """
    Train until the validation loss stops improving, then go back to the best
    weights. This is the single most useful protection against overfitting.
    """
    import keras

    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[stopper], verbose=verbose,
    )


def predict_returns(model, X):
    """Run a model and return a flat NumPy array. (Calling the model directly avoids noisy retracing warnings in loops.)"""
    return np.asarray(model(X, training=False)).reshape(-1)
