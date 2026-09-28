"""Convolutional Neural Network (CNN) Training Pipeline for Welding Defect Detection.

Constructs, compiles, and trains a multi-class CNN architecture on augmented industrial
images, outputting model weights and training convergence plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

DEFAULT_DATA_PATH: Final[str] = "Dataset_Procesado"
DEFAULT_MODEL_PATH: Final[str] = "modelo_soldadura.h5"
DEFAULT_PLOT_PATH: Final[str] = "graficas_rendimiento.png"
DOCS_ASSETS_PLOT: Final[str] = "docs/assets/graficas_rendimiento.png"
BATCH_SIZE: Final[int] = 32
IMAGE_SIZE: Final[Tuple[int, int]] = (224, 224)
INPUT_SHAPE: Final[Tuple[int, int, int]] = (224, 224, 3)
DEFAULT_EPOCHS: Final[int] = 10
RANDOM_SEED: Final[int] = 123


def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Builds the 3-block convolutional neural network architecture.

    Args:
        input_shape: Shape tuple (height, width, channels) of input tensors.
        num_classes: Total number of categorical target classes.

    Returns:
        tf.keras.Model: Compiled Sequential Keras model.
    """
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_datasets(
    data_dir: Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Loads and partitions directory images into train and validation sets (80/20).

    Args:
        data_dir: Path to directory containing class subfolders.
        image_size: Target dimension tuple (height, width).
        batch_size: Mini-batch size.
        seed: Random seed for deterministic partition.

    Returns:
        Tuple containing train_ds, val_ds, and total class count.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir.resolve()}' not found. "
            "Execute 'etapa1_augmentation.py' first to generate augmented data."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    num_classes = len(train_ds.class_names)
    return train_ds, val_ds, num_classes


def plot_training_history(history: tf.keras.callbacks.History, output_paths: list[Path]) -> None:
    """Renders and saves Accuracy and Loss curves across training epochs.

    Args:
        history: Keras History object containing epoch metrics.
        output_paths: Destination file paths for saving the plot figure.
    """
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Entrenamiento (Train)", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validación (Val)", linewidth=2)
    plt.title("Precisión (Accuracy)")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Entrenamiento (Train)", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validación (Val)", linewidth=2)
    plt.title("Pérdida (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=300)
    plt.close()


def main() -> None:
    """CLI entrypoint for running model training."""
    parser = argparse.ArgumentParser(description="Train CNN model for welding defect detection.")
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--output-model", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    try:
        train_ds, val_ds, num_classes = load_datasets(
            data_dir=args.data,
            image_size=IMAGE_SIZE,
            batch_size=args.batch_size,
        )

        print(f"[INFO] Classes detected ({num_classes}): {train_ds.class_names}")
        print(f"[INFO] Building CNN architecture with {num_classes} outputs...")

        model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=num_classes)
        model.summary()

        print(f"\n[INFO] Starting model training for {args.epochs} epochs...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

        plot_targets = [Path(DEFAULT_PLOT_PATH), Path(DOCS_ASSETS_PLOT)]
        plot_training_history(history, plot_targets)
        print(f"[INFO] Performance plots saved to: {[str(p) for p in plot_targets]}")

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output_model))
        print(f"[SUCCESS] Model successfully saved to '{args.output_model.resolve()}'.")

    except Exception as exc:
        print(f"[ERROR] Training execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()