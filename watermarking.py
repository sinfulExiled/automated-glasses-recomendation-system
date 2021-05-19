import cv2
import numpy as np


def transparentOverlay(src, overlay, x, y, scale=1):
    src = src.copy()
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if y + i >= rows or x + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[y + i][x + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[y + i][x + j]
    return src


def watermarking(original, watermarked, alpha=1, x=0, y=0):
    overlay = transparentOverlay(original, watermarked, x, y)
    output = original.copy()
    cv2.addWeighted(overlay, 1, output, 1 - 1, 0, output)
    return output
