"""Threshold / blob LED detection with optional subpixel Gaussian fit."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import optimize


@dataclass
class LEDDetection:
    u: float
    v: float
    brightness: float
    area_px: float
    confidence: float
    valid: bool


def _compactness(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri < 1e-6:
        return 0.0
    return 4.0 * np.pi * area / (peri * peri)


def detect_led_frame(
    bgr: np.ndarray,
    *,
    grayscale: bool = True,
    channel: str = "g",
    threshold: int = 240,
    min_area_px: float = 3.0,
    max_area_px: float = 50000.0,
    morphology_open: int = 0,
    morphology_close: int = 0,
    subpixel_gaussian_fit: bool = False,
    hsv_lower: np.ndarray | None = None,
    hsv_upper: np.ndarray | None = None,
) -> LEDDetection:
    """
    Detect brightest blob; return centroid and confidence.
    Missing detection → valid=False with NaNs for u,v.
    """
    if hsv_lower is not None and hsv_upper is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        gray = mask
    elif grayscale:
        if channel.lower() == "g":
            gray = bgr[:, :, 1]
        elif channel.lower() == "r":
            gray = bgr[:, :, 2]
        elif channel.lower() == "b":
            gray = bgr[:, :, 0]
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if morphology_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_open, morphology_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if morphology_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_close, morphology_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return LEDDetection(np.nan, np.nan, 0.0, 0.0, 0.0, False)

    best = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area_px or area > max_area_px:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        u = float(M["m10"] / M["m00"])
        v = float(M["m01"] / M["m00"])
        mask_roi = np.zeros_like(binary)
        cv2.drawContours(mask_roi, [c], -1, 255, -1)
        brightness = float(np.mean(gray[mask_roi > 0])) if np.any(mask_roi) else 0.0
        comp = _compactness(c)
        score = area * brightness * (0.5 + 0.5 * comp)
        if score > best_score:
            best_score = score
            best = (u, v, area, brightness, comp, c)

    if best is None:
        return LEDDetection(np.nan, np.nan, 0.0, 0.0, 0.0, False)

    u, v, area, brightness, comp, contour = best
    conf = float(np.clip(np.log1p(area) * brightness / 255.0 * comp, 0.0, 1.0))

    if subpixel_gaussian_fit:
        x, y, w, h = cv2.boundingRect(contour)
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        patch = gray[y0:y1, x0:x1].astype(np.float64)
        if patch.size < 4:
            return LEDDetection(u, v, brightness, area, conf, True)

        yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
        amp = float(patch.max() - patch.min() + 1e-6)
        bg = float(patch.min())

        def gauss(p):
            cx, cy, sx, sy = p
            g = bg + amp * np.exp(
                -0.5 * (((xx - cx) / (sx + 0.5)) ** 2 + ((yy - cy) / (sy + 0.5)) ** 2)
            )
            return np.sum((patch - g) ** 2)

        cx0 = float(np.argmax(patch.max(axis=1)))
        cy0 = float(np.argmax(patch.max(axis=0)))
        res = optimize.minimize(
            gauss,
            x0=[cx0, cy0, 2.0, 2.0],
            method="L-BFGS-B",
            bounds=[(0, patch.shape[0]), (0, patch.shape[1]), (0.5, 20), (0.5, 20)],
        )
        if res.success:
            u = float(x0 + res.x[1])
            v = float(y0 + res.x[0])

    return LEDDetection(u, v, brightness, area, conf, True)
