"""Data Augmentation Pipeline for Industrial Steel Welding Defect Classification.

Applies geometric and photometric transformations—specifically Gaussian noise—to
simulate hostile industrial plant conditions (sensor noise, lighting shifts, and dust)
and balance minority defect classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final, List

import albumentations as A
import cv2
from tqdm import tqdm

DEFAULT_SOURCE_DIR: Final[str] = "Maestria"
DEFAULT_TARGET_DIR: Final[str] = "Dataset_Procesado"
DEFAULT_VARIATIONS: Final[int] = 40
IMAGE_SIZE: Final[int] = 224


def build_augmentation_pipeline(img_size: int = IMAGE_SIZE) -> A.Compose:
    """Builds an Albumentations composition pipeline.

    Args:
        img_size: Target image height and width for model input.

    Returns:
        A.Compose: The configured augmentation pipeline.
    """
    return A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        # Industrial camera sensor noise and illumination variation simulation
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Rotate(limit=20, p=0.5),
    ])


def augment_class_images(
    source_dir: Path,
    target_dir: Path,
    variations_per_image: int,
    pipeline: A.Compose,
) -> int:
    """Executes data augmentation on all defect classes found in source_dir.

    Args:
        source_dir: Root directory containing raw images separated by class subdirectories.
        target_dir: Output directory where augmented images are saved.
        variations_per_image: Number of synthetic variants generated per original sample.
        pipeline: Albumentations transformation pipeline.

    Returns:
        int: Total number of synthetic images generated.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir.resolve()}")

    target_dir.mkdir(parents=True, exist_ok=True)
    classes: List[Path] = [p for p in source_dir.iterdir() if p.is_dir()]

    if not classes:
        raise ValueError(f"No class subdirectories found in: {source_dir.resolve()}")

    total_generated: int = 0

    for class_path in classes:
        class_name: str = class_path.name
        class_target_dir: Path = target_dir / class_name
        class_target_dir.mkdir(parents=True, exist_ok=True)

        image_files: List[Path] = [
            f for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        if not image_files:
            continue

        print(f"[INFO] Augmenting class '{class_name}' ({len(image_files)} source images)...")

        for img_file in tqdm(image_files, desc=f"Processing {class_name}", unit="img"):
            image_bgr = cv2.imread(str(img_file))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            stem_name = img_file.stem

            for idx in range(variations_per_image):
                augmented = pipeline(image=image_rgb)["image"]
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                out_path = class_target_dir / f"{stem_name}_v{idx}.jpg"
                cv2.imwrite(str(out_path), augmented_bgr)
                total_generated += 1

    return total_generated


def main() -> None:
    """CLI entrypoint for running the augmentation stage."""
    parser = argparse.ArgumentParser(
        description="Generate augmented dataset for industrial welding defect classification."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(DEFAULT_SOURCE_DIR),
        help=f"Path to original dataset (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(DEFAULT_TARGET_DIR),
        help=f"Path to destination dataset (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=DEFAULT_VARIATIONS,
        help=f"Number of augmented variants per sample (default: {DEFAULT_VARIATIONS})",
    )

    args = parser.parse_args()

    try:
        pipeline = build_augmentation_pipeline(IMAGE_SIZE)
        generated_count = augment_class_images(
            source_dir=args.source,
            target_dir=args.target,
            variations_per_image=args.variations,
            pipeline=pipeline,
        )
        print(f"\n[SUCCESS] Generated {generated_count} augmented images in '{args.target.resolve()}'.")
    except Exception as exc:
        print(f"[ERROR] Augmentation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()