"""Compute pixel-level F1 and IoU for binary prediction masks."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_binary(path: Path, threshold: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > threshold


def counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} versus {target.shape}")
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, np.logical_not(target)).sum())
    false_negative = int(np.logical_and(np.logical_not(prediction), target).sum())
    return true_positive, false_positive, false_negative


def metrics(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float]:
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    f1 = 1.0 if f1_denominator == 0 else 2 * true_positive / f1_denominator
    iou = 1.0 if iou_denominator == 0 else true_positive / iou_denominator
    return f1, iou


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare predicted PNG masks with MPLD binary ground truth."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=127)
    args = parser.parse_args()

    prediction_paths = sorted(args.predictions.glob("*.png"))
    if not prediction_paths:
        raise FileNotFoundError(f"No PNG files found in {args.predictions}")

    per_image = []
    totals = np.zeros(3, dtype=np.int64)
    for prediction_path in prediction_paths:
        target_path = args.ground_truth / prediction_path.name
        if not target_path.exists():
            raise FileNotFoundError(f"Missing ground truth: {target_path}")
        result = counts(
            load_binary(prediction_path, args.threshold),
            load_binary(target_path, args.threshold),
        )
        totals += result
        per_image.append(metrics(*result))

    macro_f1, macro_iou = np.mean(per_image, axis=0)
    micro_f1, micro_iou = metrics(*totals.tolist())
    print(f"Images: {len(per_image)}")
    print(f"Macro F1: {macro_f1:.6f}")
    print(f"Macro IoU: {macro_iou:.6f}")
    print(f"Micro F1: {micro_f1:.6f}")
    print(f"Micro IoU: {micro_iou:.6f}")


if __name__ == "__main__":
    main()
