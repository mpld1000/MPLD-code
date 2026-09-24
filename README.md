# MPLD data processing code

This repository contains only the custom scripts required to reproduce the
binary mask representation and calculate pixel-level F1 and IoU for MPLD.

## Generate binary masks

The released masks follow the LabelMe v4.6.0 `linestrip` rasterization rule:
a 10-pixel-wide line is drawn for each ordered point sequence, and all line
pixels are merged into one single-channel mask with background value 0 and
power-line value 255.

```bash
python generate_binary_masks.py \
  --annotations /path/to/annotations \
  --output /path/to/binary_masks
```

## Evaluate binary predictions

Prediction and ground-truth PNG files must have identical filenames and image
dimensions. Pixels greater than 127 are treated as foreground.

```bash
python evaluate_binary_masks.py \
  --predictions /path/to/predictions \
  --ground-truth /path/to/binary_masks
```

The script reports both macro averages over images and micro scores calculated
from pixel counts pooled across the dataset.

## Dependencies

```bash
pip install -r requirements.txt
```

LabelMe v4.6.0 is cited because its default 10-pixel `linestrip` rasterization
rule was used for the released masks.
