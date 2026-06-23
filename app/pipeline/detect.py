"""Cover isolation: find the book in the frame, deskew it, resize for the LLM.

A clean, deskewed crop materially improves recognition on angled/cluttered phone
photos. The approach mirrors the existing kinseb pipeline's geometry (largest
4-point contour → perspective de-warp) but is generic — no per-book logic. If no
confident cover region is found we gracefully fall back to the full frame; the
vision model is robust enough that this is a safe default, never an error.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..config import Config

log = logging.getLogger(__name__)


def detect_cover(image_bytes: bytes, cfg: Config) -> tuple[bytes, dict]:
    """Return (jpeg_bytes_of_cover, detection_metadata)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Unreadable bytes — hand the original through; the provider sniffs type.
        return image_bytes, {"method": "passthrough", "note": "cv2 could not decode"}

    h, w = img.shape[:2]
    meta: dict = {"method": "full_frame", "source_size": [w, h]}

    quad = _largest_quad(img)
    if quad is not None:
        area_ratio = _quad_area(quad) / float(w * h)
        meta["area_ratio"] = round(area_ratio, 3)
        # Only warp when the quad is a meaningful sub-region (a real crop), not
        # the whole frame and not a tiny spurious contour.
        if 0.18 <= area_ratio <= 0.97:
            img = _warp(img, quad)
            meta["method"] = "contour_warp"

    img = _resize(img, cfg.max_image_dim)
    if cfg.enhance_cover:
        img = _enhance(img)
        meta["enhanced"] = True

    out_h, out_w = img.shape[:2]
    meta["output_size"] = [out_w, out_h]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return image_bytes, {"method": "passthrough", "note": "jpeg encode failed"}
    return buf.tobytes(), meta


def _largest_quad(img: np.ndarray) -> np.ndarray | None:
    """Find the largest convex 4-point contour (the cover rectangle)."""
    scale = 800.0 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: np.ndarray | None = None
    best_area = 0.0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area > best_area:
            best_area = area
            best = approx
    if best is None:
        return None
    inv = 1.0 / scale if scale < 1 else 1.0
    return (best.reshape(4, 2).astype(np.float32) * inv)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(quad.astype(np.float32)))


def _warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def _resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _enhance(img: np.ndarray) -> np.ndarray:
    """Mild CLAHE contrast boost on the luminance channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
