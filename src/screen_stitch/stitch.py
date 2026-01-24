from dataclasses import dataclass
import numpy as np
import cv2
from screen_stitch.cvutils import (
    to_gray,
    estimate_scroll_phase_corr,
    normalized_cross_correlation,
    estimate_scroll_template,
)
from screen_stitch.layout_detection import Layout
from pathlib import Path


@dataclass
class StitchParams:
    phase_max_dx_allowed_px: int
    phase_min_response: float
    phase_min_overlap_ncc: float
    template_min_score: float
    min_scroll_frac: int


def append_strip(strips: list[np.ndarray], roi_bgr: np.ndarray, scroll_px: int) -> None:
    h = roi_bgr.shape[0]
    scroll_px = int(max(0, min(h, scroll_px)))
    if scroll_px <= 0:
        return
    strips.append(roi_bgr[h - scroll_px : h, :, :])


def determine_scroll_px_by_phase(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    params: StitchParams,
) -> int | None:
    dx, dy, resp = estimate_scroll_phase_corr(prev_gray, cur_gray)
    dy = int(round(-dy))
    if abs(dx) > params.phase_max_dx_allowed_px:
        return None
    if dy <= 0 or dy >= cur_gray.shape[0]:
        return None
    if resp < params.phase_min_response:
        return None
    overlap_h = cur_gray.shape[0] - dy
    if overlap_h < 20:
        return None
    ncc = normalized_cross_correlation(prev_gray[dy:, :], cur_gray[:overlap_h, :])
    if ncc >= params.phase_min_overlap_ncc:
        return dy
    else:
        return None


def determine_scroll_px_by_template(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    params: StitchParams,
) -> int | None:
    result = estimate_scroll_template(
        prev_gray, cur_gray, tmpl_h=int(params.min_scroll_frac * prev_gray.shape[0])
    )
    if result is None:
        return None
    scroll_px, score = result
    if score < params.template_min_score:
        return None
    else:
        return scroll_px


def determine_scroll_px(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    params: StitchParams,
) -> int | None:
    scroll_px = determine_scroll_px_by_phase(prev_gray, cur_gray, params)
    if scroll_px is None:
        scroll_px = determine_scroll_px_by_template(prev_gray, cur_gray, params)
    return scroll_px


def stitch_video(
    video_path: Path,
    layout: Layout,
    params: StitchParams,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    idx = -1
    roi_start, roi_end = layout.header_row_incl[1] + 1, layout.footer_row
    min_scroll_px = int((roi_end - roi_start) * params.min_scroll_frac)
    strips = []
    prev_gray = None
    used_frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        if idx < layout.start_frame_idx:
            continue

        if len(strips) == 0:
            strips.append(frame[roi_start:roi_end, :, :])
            used_frames.append(idx)
            prev_gray = to_gray(strips[-1])
            continue

        cur_roi = frame[roi_start:roi_end, :, :]
        cur_gray = to_gray(cur_roi)
        scroll_px = determine_scroll_px(prev_gray, cur_gray, params)
        if scroll_px is None or scroll_px < min_scroll_px:
            continue

        append_strip(strips, cur_roi, scroll_px)
        prev_gray = cur_gray
        used_frames.append(idx)

    cap.release()

    # handle last frame
    scroll_px = determine_scroll_px(prev_gray, cur_gray, params)
    if scroll_px is not None:
        # NOTE optionally to use 0.5x phase response limit and 0.85x ncc to be more lenient
        append_strip(strips, cur_roi, scroll_px)
        used_frames.append(idx)

    strips = [layout.header_frame] + strips
    ret = np.vstack(strips)
    print(f"Used {len(used_frames)} frames: {used_frames}, final shape {ret.shape}")
    return ret
