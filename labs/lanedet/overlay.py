"""Block 6 - overlay: draw what the detector saw on top of the frame."""

import math

import cv2
import numpy as np

CYAN, ORANGE, YELLOW, WHITE, GREY = (0, 220, 255), (255, 150, 0), (255, 235, 0), (240, 240, 240), (120, 120, 120)


def draw(frame: np.ndarray, det, detector, scale: int = 4) -> np.ndarray:
    """Frame | mask side by side, upscaled, with bands, blobs, centre points, centreline and a status line."""
    H, W = frame.shape[:2]
    cam = cv2.resize(frame, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    mask_rgb = (np.stack([det.mask] * 3, -1) * 255).astype(np.uint8)
    msk = cv2.resize(mask_rgb, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    s = scale

    pts = []
    for img in (cam, msk):
        cv2.line(img, (W * s // 2, 0), (W * s // 2, H * s), GREY, 1)
        for est, rows, colour in ((det.near, detector.near_rows, CYAN), (det.far, detector.far_rows, ORANGE)):
            cv2.rectangle(img, (0, rows[0] * s), (W * s - 1, rows[1] * s), colour, 1)
            v = detector.band_row(rows)
            for line in est.used:
                cv2.drawMarker(img, (int(line.u * s), int(v * s)), WHITE, cv2.MARKER_TILTED_CROSS, 10, 2)
            if est.u is not None:
                cv2.circle(img, (int(est.u * s), int(v * s)), 5, colour, -1)
        x_n, x_f = det.x_near, det.x_far
        v_n, v_f = detector.band_row(detector.near_rows), detector.band_row(detector.far_rows)
        pts = [(int(detector.camera.to_pixel_u(x, v) * s), int(v * s)) for x, v in ((x_n, v_n), (x_f, v_f))]
        cv2.line(img, pts[0], pts[1], YELLOW, 2, cv2.LINE_AA)

    strip = np.full((26, W * s * 2, 3), 20, dtype=np.uint8)
    text = (
        f"e_lat {det.e_lat:+.2f}   e_heading {math.degrees(det.e_heading):+.1f} deg   "
        f"conf {det.confidence:.2f}   near {det.near.case}   far {det.far.case}"
    )
    cv2.putText(strip, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([np.hstack([cam, msk]), strip])
