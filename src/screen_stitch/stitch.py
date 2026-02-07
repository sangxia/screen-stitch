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
    min_scroll_frac: float


def append_strip(
    strips: list[np.ndarray],
    roi_bgr: np.ndarray,
    scroll_px: int,
    idx: int | None = None,
) -> None:
    """Append the newly revealed strip from a scrolling ROI.

    Args:
        strips: Accumulated list of vertical strips for stitching. Each entry
            has shape (H, W, 3) with dtype ``uint8``.
        roi_bgr: Current frame region-of-interest in BGR with shape (H, W, 3)
            and dtype ``uint8``.
        scroll_px: The height of the bottom strip in ``roi_bgr`` to be cropped.
    """
    assert 0 <= scroll_px < roi_bgr.shape[0]
    if scroll_px:
        img = roi_bgr[-scroll_px:, :, :].copy()
        # # NOTE uncomment to add ID for debugging
        # if idx is not None:
        #     img = cv2.putText(
        #         img, str(idx), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1
        #     )
        strips.append(img)


def determine_scroll_px_by_phase(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    params: StitchParams,
) -> int | None:
    """Estimate scroll using phase correlation with overlap verification.

    This first uses phase correlation to estimate vertical translation, then
    checks horizontal drift, response strength, and overlap similarity via
    normalized cross-correlation to reject unreliable matches.

    Args:
        prev_gray: Previous grayscale ROI with shape (H, W) and dtype
            ``uint8`` (or compatible numeric type).
        cur_gray: Current grayscale ROI with shape (H, W) and dtype matching
            ``prev_gray``.
        params: Stitching parameters controlling thresholds.

    Returns:
        The estimated vertical scroll in pixels, or ``None`` if invalid.
    """
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
    """Estimate scroll via template matching when phase correlation fails.

    Args:
        prev_gray: Previous grayscale ROI with shape (H, W) and dtype
            ``uint8`` (or compatible numeric type).
        cur_gray: Current grayscale ROI with shape (H, W) and dtype matching
            ``prev_gray``.
        params: Stitching parameters controlling template thresholds.

    Returns:
        The estimated vertical scroll in pixels, or ``None`` if invalid.
    """
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
    """Select the best scroll estimate using phase correlation then template matching.

    Args:
        prev_gray: Previous grayscale ROI with shape (H, W) and dtype
            ``uint8`` (or compatible numeric type).
        cur_gray: Current grayscale ROI with shape (H, W) and dtype matching
            ``prev_gray``.
        params: Stitching parameters controlling thresholds.

    Returns:
        The estimated vertical scroll in pixels, or ``None`` if no method passes.
    """
    scroll_px = determine_scroll_px_by_phase(prev_gray, cur_gray, params)
    if scroll_px is None:
        scroll_px = determine_scroll_px_by_template(prev_gray, cur_gray, params)
    return scroll_px


def stitch_video(
    video_path: Path,
    layout: Layout,
    params: StitchParams,
) -> np.ndarray:
    """Stitch a vertically scrolling video into a single tall image.

    This extracts a stable region-of-interest between header and footer, then
    (optionally) collects carousel frames before estimating vertical translation
    between consecutive frames using phase correlation or template matching.
    Newly revealed strips are appended and stacked to create the final stitched
    image.

    Args:
        video_path: Path to the input video.
        layout: Detected layout bounds for header/footer.
        params: Stitching parameters controlling thresholds.

    Returns:
        The stitched image as a single BGR array with shape (H_total, W, 3) and
        dtype ``uint8``.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    idx = -1
    roi_start, roi_end = layout.header_row_incl[1] + 1, layout.footer_row
    min_scroll_px = int((roi_end - roi_start) * params.min_scroll_frac)
    strips = []
    cur_gray = None
    prev_gray = None
    used_frames = []
    carousel_row_end = layout.carousel_last_row
    carousel_frame_indices = layout.carousel_frame_indices
    carousel_end_idx = layout.carousel_end_frame_idx
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        if idx < layout.start_frame_idx:
            continue
        if carousel_end_idx is not None and idx <= carousel_end_idx:
            # Collect carousel images before the scrolling portion begins.
            if idx in carousel_frame_indices:
                strips.append(frame[roi_start : carousel_row_end + 1, :, :])
                used_frames.append(idx)
            elif idx == carousel_end_idx:
                # Transition frame: add the non-carousel remainder and seed scrolling state.
                if carousel_row_end + 1 < roi_end:
                    strips.append(frame[carousel_row_end + 1 : roi_end, :, :])
                prev_gray = to_gray(frame[roi_start:roi_end, :, :])
                used_frames.append(idx)
            continue

        if len(strips) == 0:
            # Seed the scroll ROI if there was no carousel segment.
            strips.append(frame[roi_start:roi_end, :, :])
            used_frames.append(idx)
            prev_gray = to_gray(strips[-1])
            continue

        cur_roi = frame[roi_start:roi_end, :, :]
        cur_gray = to_gray(cur_roi)
        # Estimate how much the content scrolled between frames.
        scroll_px = determine_scroll_px(prev_gray, cur_gray, params)
        if scroll_px is None or scroll_px < min_scroll_px:
            continue

        append_strip(strips, cur_roi, scroll_px, idx=idx)
        prev_gray = cur_gray
        used_frames.append(idx)

    cap.release()
    if prev_gray is None:
        raise RuntimeError("Unable to find any frames, start index likely incorrect.")

    # handle last frame
    if cur_gray is not None:
        scroll_px = determine_scroll_px(prev_gray, cur_gray, params)
        if scroll_px is not None:
            # NOTE optionally to use 0.5x phase response limit and 0.85x ncc to be more lenient
            append_strip(strips, cur_roi, scroll_px)
            used_frames.append(idx)

    strips = [layout.header_frame] + strips
    ret = np.vstack(strips)
    print(f"Used {len(used_frames)} frames: {used_frames}, final shape {ret.shape}")
    return ret
