"""
FER (Facial Emotion Recognition) - Training Script

Trains a lightweight residual depthwise-separable CNN (mini-Xception family)
on FER2013. 48x48 grayscale input, built for real-time inference.

Usage:
    python Main.py
    python Main.py --epochs 200 --batch-size 128
    python Main.py --data-path data/fer2013.csv
    python Main.py --ferplus-path data/fer2013new.csv   # use FERPlus labels
    python Main.py --quick-test   # tiny run to sanity-check the pipeline
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import json
import random
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers, losses

SEED = 42
IMG_SIZE = 48
NUM_CLASSES = 7
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

MODELS_DIR = "models"
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():
    """Prevents TF from allocating all GPU memory up front."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"GPUs detected: {len(gpus)}")


FERPLUS_VOTE_COLUMNS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt", "unknown", "NF",
]

# Map FERPlus columns to original 7 classes. 
# 'Contempt' is merged into 'Neutral' to maintain output shape.
FERPLUS_TO_CLASS_IDX = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3,
    "sadness": 4, "surprise": 5, "neutral": 6, "contempt": 6,
}


def load_ferplus_labels(ferplus_path, n_rows):
    """
    Loads FERPlus majority-vote labels. Returns (labels, keep_mask), 
    where keep_mask drops rows flagged as 'unknown' or 'not a face'.
    """
    fdf = pd.read_csv(ferplus_path)
    fdf.columns = [c.strip() for c in fdf.columns]

    missing = set(FERPLUS_VOTE_COLUMNS) - set(fdf.columns)
    if missing:
        raise ValueError(f"fer2013new.csv is missing expected columns: {missing}")
    if len(fdf) != n_rows:
        raise ValueError(
            f"fer2013new.csv has {len(fdf)} rows but fer2013.csv has {n_rows}. "
            "They must be the same FER2013 release, row-aligned."
        )

    votes = fdf[FERPLUS_VOTE_COLUMNS].values.astype("int64")
    majority_idx = votes.argmax(axis=1)
    majority_name = np.array(FERPLUS_VOTE_COLUMNS)[majority_idx]

    keep_mask = ~np.isin(majority_name, ["unknown", "NF"])
    labels = np.array([FERPLUS_TO_CLASS_IDX.get(name, -1) for name in majority_name], dtype="int64")

    return labels, keep_mask


def load_fer2013(csv_path, ferplus_path=None):
    """
    Vectorized loader returning pixels (N, 48, 48, 1), labels, and Usage array.
    Uses FERPlus majority-vote labels if provided.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"emotion", "pixels", "Usage"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    pixels = df["pixels"].apply(lambda s: np.array(s.split(), dtype="uint8"))
    bad = pixels.apply(len) != IMG_SIZE * IMG_SIZE
    if bad.any():
        raise ValueError(f"{bad.sum()} rows do not have {IMG_SIZE*IMG_SIZE} pixel values.")

    X = np.stack(pixels.values).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    usage = df["Usage"].values

    if ferplus_path:
        y, keep_mask = load_ferplus_labels(ferplus_path, n_rows=len(df))
        dropped = (~keep_mask).sum()
        print(
            f"Using FERPlus labels: dropped {dropped} rows "
            f"({dropped / len(df):.1%}) flagged 'unknown'/'not a face'."
        )
        X, y, usage = X[keep_mask], y[keep_mask], usage[keep_mask]
    else:
        y = df["emotion"].values.astype("int64")

    return X, y, usage


def split_by_usage(X, y, usage):
    train_mask = usage == "Training"
    val_mask = usage == "PublicTest"
    test_mask = usage == "PrivateTest"

    splits = {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }
    for name, (Xs, ys) in splits.items():
        if len(Xs) == 0:
            raise ValueError(
                f"'{name}' split is empty. Check the 'Usage' column values in your CSV."
            )
    return splits


def compute_class_weights(y_train):
    """
    Applies softened (sqrt) inverse-frequency weights to boost rare classes 
    without destroying precision.
    """
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype("float64")
    counts[counts == 0] = 1
    total = counts.sum()
    raw_weights = total / (NUM_CLASSES * counts)
    weights = np.sqrt(raw_weights)
    return {i: float(w) for i, w in enumerate(weights)}


_AUGMENTATION = None


def get_augmentation_layer():
    """Built lazily and kept on CPU to prevent GPU pipeline stalling."""
    global _AUGMENTATION
    if _AUGMENTATION is None:
        _AUGMENTATION = models.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomTranslation(0.1, 0.1),
                layers.RandomZoom(0.1),
                layers.GaussianNoise(8.0),
            ],
            name="augmentation",
        )
    return _AUGMENTATION


def preprocess(x, y, augment=False):
    """Scales pixels to [-1, 1], applies CPU-side augmentation, and one-hot encodes labels."""
    x = tf.cast(x, tf.float32)
    if augment:
        x = get_augmentation_layer()(x, training=True)
    x = (x / 255.0 - 0.5) * 2.0
    y = tf.one_hot(y, NUM_CLASSES)
    return x, y


def make_dataset(X, y, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    with tf.device("/CPU:0"):
        ds = ds.map(lambda x, y: preprocess(x, y, augment=augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES, l2_reg=5e-3):
    reg = regularizers.l2(l2_reg)

    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(8, 3, use_bias=False, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(8, 3, use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual depthwise-separable blocks
    for filters in (16, 32, 64, 128):
        residual = layers.Conv2D(filters, 1, strides=2, padding="same", use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(
            filters, 3, padding="same", use_bias=False,
            depthwise_regularizer=reg, pointwise_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(
            filters, 3, padding="same", use_bias=False,
            depthwise_regularizer=reg, pointwise_regularizer=reg,
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, residual])

    x = layers.Conv2D(num_classes, 3, padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax", name="predictions")(x)

    return models.Model(inputs, outputs, name="fer_mini_xception")


def build_callbacks(checkpoint_path, log_path):
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=25, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        callbacks.CSVLogger(log_path),
    ]


def evaluate_and_report(model, test_ds, y_test_raw, report_path):
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nPrivateTest (held-out) — loss: {loss:.4f}  accuracy: {acc:.4f}")

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    lines = [f"PrivateTest loss: {loss:.4f}", f"PrivateTest accuracy: {acc:.4f}", ""]

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        class_ids = list(range(NUM_CLASSES))
        report = classification_report(
            y_test_raw, y_pred, labels=class_ids, target_names=CLASS_NAMES, digits=4, zero_division=0
        )
        cm = confusion_matrix(y_test_raw, y_pred, labels=class_ids)
        lines.append("Classification report:\n" + report)
        lines.append("Confusion matrix (rows=true, cols=pred):\n" + np.array2string(cm))
        print(report)
    except ImportError:
        lines.append(
            "scikit-learn not installed — skipped per-class report. "
            "Install with: pip install scikit-learn"
        )
        print("scikit-learn not installed — skipped per-class report.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def save_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Legacy json + h5 format for backwards compatibility
    with open(os.path.join(MODELS_DIR, "model.json"), "w") as f:
        f.write(model.to_json())

    weights_path = os.path.join(MODELS_DIR, "model.h5")
    try:
        model.save_weights(weights_path)
    except ValueError:
        # Keras 3 format handling. App.py will require updates if TF is upgraded.
        tmp_path = os.path.join(MODELS_DIR, "model.weights.h5")
        model.save_weights(tmp_path)
        import shutil
        shutil.copyfile(tmp_path, weights_path)
        print(
            "NOTE: saved with Keras 3's native weights format. "
            "app.py's model_from_json()+load_weights() expects the legacy "
            "Keras 2 format and will need updating to match."
        )

    # Modern single-file format
    model.save(os.path.join(MODELS_DIR, "model.keras"))

    with open(os.path.join(MODELS_DIR, "class_labels.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f"\nSaved model.json + model.h5 + model.keras + class_labels.json to '{MODELS_DIR}/'")


def main():
    parser = argparse.ArgumentParser(description="Train the FER model.")
    parser.add_argument("--data-path", default=os.path.join("data", "fer2013.csv"))
    parser.add_argument(
        "--ferplus-path",
        default=None,
        help="Path to fer2013new.csv. If given, uses FERPlus's crowd-sourced "
             "majority-vote labels instead of the original single-annotator "
             "labels, and drops rows flagged unknown/not-a-face.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run 1 epoch on a tiny model to sanity-check the pipeline end to end.",
    )
    args = parser.parse_args()

    set_seeds()
    configure_gpu()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading dataset from '{args.data_path}' ...")
    X, y, usage = load_fer2013(args.data_path, ferplus_path=args.ferplus_path)
    splits = split_by_usage(X, y, usage)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (
        splits["train"],
        splits["val"],
        splits["test"],
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    class_weight = compute_class_weights(y_train)
    print("Class weights:", {CLASS_NAMES[k]: round(v, 2) for k, v in class_weight.items()})

    batch_size = 4 if args.quick_test else args.batch_size
    epochs = 1 if args.quick_test else args.epochs

    train_ds = make_dataset(X_train, y_train, batch_size, shuffle=True, augment=True)
    val_ds = make_dataset(X_val, y_val, batch_size)
    test_ds = make_dataset(X_test, y_test, batch_size)

    model = build_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.summary()

    cbs = build_callbacks(
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "best_model.weights.h5"),
        log_path=os.path.join(MODELS_DIR, "training_log.csv"),
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )

    evaluate_and_report(
        model, test_ds, y_test, os.path.join(MODELS_DIR, "evaluation_report.txt")
    )
    save_model(model)


if __name__ == "__main__":
    main()