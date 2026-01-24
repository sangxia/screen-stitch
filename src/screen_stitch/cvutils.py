import cv2
import numpy as np


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-6
    return float((a * b).sum() / denom)


def maybe_resize_gray_by_scale(g: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize grayscale image so max(h,w) <= max_dim. Returns (resized, scale)."""
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
    """Return (dx, dy, response) using phase correlation."""
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
    """Fallback: match bottom strip of prev inside cur, return (scroll_px, score)."""
    h, w = prev_gray.shape
    res = cv2.matchTemplate(cur_gray, prev_gray[-tmpl_h:, :], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    y = max_loc[1]
    scroll = (h - tmpl_h) - y
    if scroll <= 0 or scroll >= h:
        return None
    return int(scroll), float(max_val)
