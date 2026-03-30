import numpy as np
import cv2

from face_mc_gs.sync.led_detector import detect_led_frame


def test_bright_blob():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(img, (80, 60), 8, (0, 255, 0), -1)
    det = detect_led_frame(img, threshold=200, grayscale=True, channel="g")
    assert det.valid
    assert abs(det.u - 80) < 2
    assert abs(det.v - 60) < 2
