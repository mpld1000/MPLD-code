"""Generate MPLD binary masks from LabelMe v4.6.0 linestrip annotations."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def annotation_to_mask(annotation_path: Path, line_width: int = 10) -> Image.Image:
    """Rasterize all linestrip objects to a single-channel 0/255 mask."""
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in annotation.get("shapes", []):
        shape_type = shape.get("shape_type")
        if shape_type not in (None, "linestrip"):
            continue
        points = [tuple(point) for point in shape.get("points", [])]
        if len(points) >= 2:
            # This matches LabelMe v4.6.0 shape_to_mask for a linestrip.
            draw.line(points, fill=255, width=line_width)

    values = np.asarray(mask)
    if not np.isin(values, [0, 255]).all():
        raise ValueError(f"Non-binary values generated for {annotation_path}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 0/255 PNG masks from MPLD LabelMe JSON files."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--line-width", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    annotation_paths = sorted(args.annotations.glob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No JSON files found in {args.annotations}")

    for annotation_path in annotation_paths:
        mask = annotation_to_mask(annotation_path, args.line_width)
        mask.save(args.output / f"{annotation_path.stem}.png")

    print(f"Generated {len(annotation_paths)} masks in {args.output}")


if __name__ == "__main__":
    main()
