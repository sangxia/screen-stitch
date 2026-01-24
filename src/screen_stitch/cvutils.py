import cv2
import numpy as np


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a single-channel grayscale image.

    Args:
        bgr: Input image in BGR channel order with shape (H, W, 3) and dtype
            ``uint8`` (or another OpenCV-compatible image dtype). The last
            dimension represents color channels.

    Returns:
        Grayscale image with shape (H, W) and dtype ``uint8`` (matching OpenCV's
        conversion output).
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the normalized cross-correlation (NCC) between two images.

    This treats the inputs as flattened vectors, subtracts the mean from each,
    and computes the cosine similarity. NCC is commonly used as a similarity
    metric for overlapping image regions in CV pipelines.

    Args:
        a: First image or patch with shape (H, W) or any matching shape; values
            are interpreted as scalar intensities and are cast to ``float32``.
        b: Second image or patch with the same shape as ``a`` and compatible
            numeric dtype.

    Returns:
        The normalized cross-correlation score in [-1, 1].
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-6
    return float((a * b).sum() / denom)


def maybe_resize_gray_by_scale(g: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize a grayscale image to keep the longest side under a limit.

    Args:
        g: Grayscale image with shape (H, W) and dtype ``uint8`` (or another
            OpenCV-compatible dtype).
        max_dim: Maximum allowed size for the height or width.

    Returns:
        A tuple of ``(resized_image, scale)``. ``resized_image`` has shape
        (H', W') with the same dtype as ``g``; ``scale`` is the factor applied
        to the original dimensions.
    """
    h, w = g.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return g, 1.0
    scale = max_dim / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def estimate_scroll_phase_corr(
    prev_gray: np.ndarray, cur_gray: np.ndarray
) -> tuple[float, float, float]:
    """Estimate translation between two frames with phase correlation.

    Phase correlation estimates the shift between two images in the frequency
    domain, which is robust to uniform illumination changes. We return the
    subpixel translation and the response score reported by OpenCV.

    Args:
        prev_gray: Previous grayscale frame with shape (H, W) and dtype
            ``uint8`` (or compatible numeric type).
        cur_gray: Current grayscale frame with shape (H, W) and dtype matching
            ``prev_gray``.

    Returns:
        A tuple of (dx, dy, response), where dx/dy are the estimated translation
        from ``prev_gray`` to ``cur_gray`` and ``response`` is a confidence-like
        score from the phase correlation peak.
    """
    # g1, s1 = maybe_resize_gray_by_scale(prev_gray, max_dim=max_dim)
    # g2, _ = maybe_resize_gray_by_scale(cur_gray, max_dim=max_dim)
    g1, g2 = prev_gray, cur_gray
    s1 = 1.0
    assert g1.shape == g2.shape
    f1 = g1.astype(np.float32)
    f2 = g2.astype(np.float32)
    win = cv2.createHanningWindow((f1.shape[1], f1.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(f1, f2, window=win)
    dx /= s1
    dy /= s1
    return float(dx), float(dy), float(resp)


def estimate_scroll_template(
    prev_gray: np.ndarray, cur_gray: np.ndarray, tmpl_h: int
) -> tuple[int, float] | None:
    """Estimate scroll by template matching a bottom strip into the next frame.

    This uses OpenCV template matching (normalized cross-correlation) to locate
    the bottom ``tmpl_h`` rows from the previous frame inside the current frame.
    It is a fallback when phase correlation is unreliable.

    Args:
        prev_gray: Previous grayscale frame with shape (H, W) and dtype
            ``uint8`` (or compatible numeric type).
        cur_gray: Current grayscale frame with shape (H, W) and dtype matching
            ``prev_gray``.
        tmpl_h: Height of the strip taken from the bottom of ``prev_gray``.

    Returns:
        ``(scroll_px, score)`` if a valid positive scroll is found, otherwise
        ``None``.
    """
    h, w = prev_gray.shape
    res = cv2.matchTemplate(cur_gray, prev_gray[-tmpl_h:, :], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    y = max_loc[1]
    scroll = (h - tmpl_h) - y
    if scroll <= 0 or scroll >= h:
        return None
    return int(scroll), float(max_val)
