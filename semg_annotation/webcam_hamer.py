#!/usr/bin/env python3
"""
Live webcam test for AnyHand-HaMeR hand pose prediction.

Run from the repo root:

    python semg_annotation/webcam_hamer.py
    python semg_annotation/webcam_hamer.py --camera 0 --every-n 2

Prerequisites:
    bash scripts/prepare_wilor.sh   # detector.pt (shared by all backends)
    bash scripts/prepare_hamer.sh   # anyhand_hamer.ckpt

Mesh overlay (--mesh) additionally requires WiLoR for the pyrender renderer;
use --mesh with backend=both (default when --mesh is set).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rgb_predictor import AnyHandPredictor, HandPrediction  # noqa: E402

# MANO 21-keypoint hand skeleton (wrist + 4 joints per finger).
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

_RIGHT_COLOR = (80, 220, 100)
_LEFT_COLOR = (80, 160, 255)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam smoke test for AnyHand-HaMeR finger pose prediction.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Requested capture width; 0 keeps the driver default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Requested capture height; 0 keeps the driver default.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Run HaMeR every N frames to reduce latency (default: 1).",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render MANO mesh overlay (loads WiLoR too; slower).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, e.g. cuda or cpu (default: auto).",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.3,
        help="YOLO hand-detector confidence threshold.",
    )
    return parser.parse_args()


def _draw_hand(
    frame: np.ndarray,
    hand: HandPrediction,
    *,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
) -> None:
    color = _RIGHT_COLOR if hand.is_right else _LEFT_COLOR
    label = f"{'R' if hand.is_right else 'L'} {hand.score:.2f}"

    if draw_bbox:
        x1, y1, x2, y2 = hand.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if not draw_skeleton:
        return

    pts = hand.keypoints_2d.astype(int)
    for i, j in _HAND_EDGES:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 24


def main() -> int:
    args = _parse_args()
    if args.every_n < 1:
        print("--every-n must be >= 1", file=sys.stderr)
        return 2

    backend = "both" if args.mesh else "hamer"
    print(f"Loading AnyHandPredictor (backend={backend!r}) …")
    predictor = AnyHandPredictor(
        backend=backend,
        device=args.device,
        det_conf=args.det_conf,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}", file=sys.stderr)
        return 1

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = "AnyHand HaMeR webcam (q/ESC to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    last_hands: list[HandPrediction] = []
    infer_ms = 0.0
    fps_smooth = 0.0
    t_prev = time.perf_counter()

    print("Press q or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.", file=sys.stderr)
                break

            run_infer = (frame_idx % args.every_n) == 0
            if run_infer:
                t0 = time.perf_counter()
                last_hands = predictor.predict(frame, backend="hamer")
                infer_ms = (time.perf_counter() - t0) * 1000.0

            display = frame.copy()
            if args.mesh and last_hands:
                display = predictor.render_overlay(frame, last_hands)
            else:
                for hand in last_hands:
                    _draw_hand(display, hand)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = inst_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst_fps

            hand_txt = "no hand"
            if last_hands:
                parts = [
                    f"{'R' if h.is_right else 'L'}:{h.score:.2f}"
                    for h in last_hands
                ]
                hand_txt = ", ".join(parts)

            _draw_hud(
                display,
                [
                    f"FPS {fps_smooth:.1f}  infer {infer_ms:.0f} ms  every {args.every_n} frame(s)",
                    f"hands: {hand_txt}",
                    "q/ESC quit",
                ],
            )

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
