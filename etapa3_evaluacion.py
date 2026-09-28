"""Evaluation and Metrics Generation for Welding Defect Classifier.

Loads trained CNN weights, evaluates inference on un-shuffled validation data,
computes multi-class confusion matrix, and outputs classification metrics report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

DEFAULT_MODEL_PATH: Final[str] = "modelo_soldadura.h5"
DEFAULT_DATA_PATH: Final[str] = "Dataset_Procesado"
DEFAULT_CM_PLOT: Final[str] = "matriz_confusion.png"
DOCS_ASSETS_CM: Final[str] = "docs/assets/matriz_confusion.png"
IMAGE_SIZE: Final[Tuple[int, int]] = (224, 224)
BATCH_SIZE: Final[int] = 32
RANDOM_SEED: Final[int] = 123


def evaluate_model(
    model_path: Path,
    data_path: Path,
    cm_output_paths: List[Path],
) -> None:
    """Executes evaluation pipeline and generates diagnostic artifacts.

    Args:
        model_path: Path to serialized Keras/H5 model weights.
        data_path: Path to dataset root directory.
        cm_output_paths: Paths where the generated confusion matrix heatmap will be written.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model '{model_path.resolve()}' not found. "
            "Execute 'etapa2_entrenamiento.py' to generate model weights."
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset '{data_path.resolve()}' not found. "
            "Execute 'etapa1_augmentation.py' to generate preprocessed images."
        )

    print(f"[INFO] Loading model weights from: {model_path.resolve()}")
    model: tf.keras.Model = tf.keras.models.load_model(str(model_path))

    # Crucial: shuffle=False ensures alignment between ground truth and predictions
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_path),
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names: List[str] = val_ds.class_names
    print(f"[INFO] Evaluating on {len(class_names)} classes: {class_names}")

    y_true = np.concatenate([y for _, y in val_ds], axis=0)
    raw_predictions = model.predict(val_ds)
    y_pred = np.argmax(raw_predictions, axis=-1)

    # 1. Confusion Matrix Generation
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Clase Predicha (Predicted)", fontsize=11)
    plt.ylabel("Clase Real (Actual)", fontsize=11)
    plt.title("Matriz de Confusión - Clasificador de Defectos en Soldadura", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for out_path in cm_output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to: {[str(p) for p in cm_output_paths]}")

    # 2. Detailed Classification Metrics Report
    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN (MÉTRICAS INDUSTRIALES)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 60)


def main() -> None:
    """CLI entrypoint for running the model evaluation stage."""
    parser = argparse.ArgumentParser(description="Evaluate welding defect classification model.")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA_PATH))
    args = parser.parse_args()

    try:
        targets = [Path(DEFAULT_CM_PLOT), Path(DOCS_ASSETS_CM)]
        evaluate_model(model_path=args.model, data_path=args.data, cm_output_paths=targets)
    except Exception as exc:
        print(f"[ERROR] Evaluation execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()