#!/usr/bin/env python3
"""Test AnyHand-HaMeR on a single image.

    python semg_annotation/test_image_hamer.py photo.jpg
    python semg_annotation/test_image_hamer.py photo.jpg -o result.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.rgb_predictor import AnyHandPredictor, HandPrediction  # noqa: E402

_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _draw_hand(frame, hand: HandPrediction) -> None:
    color = (80, 220, 100) if hand.is_right else (80, 160, 255)
    x1, y1, x2, y2 = hand.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    pts = hand.keypoints_2d.astype(int)
    for i, j in _HAND_EDGES:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, color, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test HaMeR on one image.")
    parser.add_argument("image", help="Input image path (jpg/png).")
    parser.add_argument("-o", "--out", help="Output image path (default: <name>_hamer.jpg).")
    parser.add_argument("--det-conf", type=float, default=0.3)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"File not found: {image_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else image_path.with_name(f"{image_path.stem}_hamer.jpg")

    print("Loading model …")
    predictor = AnyHandPredictor(backend="hamer", det_conf=args.det_conf)

    print(f"Predicting: {image_path}")
    hands = predictor.predict(str(image_path))
    print(f"Detected {len(hands)} hand(s)")
    for i, h in enumerate(hands):
        side = "right" if h.is_right else "left"
        print(f"  [{i}] {side}  score={h.score:.3f}")

    img = cv2.imread(str(image_path))
    for hand in hands:
        _draw_hand(img, hand)

    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
