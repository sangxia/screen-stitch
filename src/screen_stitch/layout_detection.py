from dataclasses import dataclass
import numpy as np
import cv2
from screen_stitch.cvutils import to_gray
from pathlib import Path


@dataclass
class Layout:
    start_frame_idx: int
    header_row_incl: tuple[int, int]
    footer_row: int
    header_frame: np.ndarray
    carousel_row_incl: tuple[int, int] | None = None
    carousel_end_frame_idx: int | None = None
    carousel_frame_indices: list[int] | None = None


def find_longest_segment_in_mask(mask: np.ndarray) -> tuple[int | None, int | None]:
    """Find the longest contiguous True segment in a 1D boolean mask.

    Args:
        mask: Boolean mask of shape (N,) and dtype ``bool`` indicating the rows
            that satisfy a stability criterion.

    Returns:
        A tuple of (start_index, end_index) for the longest True segment, or
        (None, None) if no True region exists.
    """
    diff_idx = np.where(mask[:-1] != mask[1:])[0] + 1
    best_start, best_end = None, None
    for seg_idx in np.split(np.arange(mask.shape[0]), diff_idx):
        if not np.all(mask[seg_idx]):
            continue
        if best_start is None or seg_idx.shape[0] > best_end - best_start + 1:
            best_start, best_end = int(seg_idx[0]), int(seg_idx[-1])
    return best_start, best_end


def find_header_stable_start(
    video_path: Path,
    max_probe_frames: int,
    top_probe_height_frac: float,
    mad_limit: float,
) -> tuple[int, int, int]:
    """Detect a stable header region based on temporal intensity variation.

    The method measures per-row mean absolute differences across frames in the
    top portion of the video to identify rows that remain visually stable
    (consistent UI chrome). It then finds the longest stable band and returns
    the earliest frame index where the header becomes steady.

    Args:
        video_path: Path to the input video.
        max_probe_frames: Maximum number of frames to analyze.
        top_probe_height_frac: Fraction of the image height to probe from the top.
        mad_limit: Mean absolute difference threshold for stability.

    Returns:
        A tuple of (start_frame_idx, header_row_start, header_row_end), where
        ``header_row_start`` and ``header_row_end`` are inclusive.

    Raises:
        RuntimeError: If no frames are available or a header cannot be found.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("No frames in video.")
    top_h = int(frame0.shape[0] * top_probe_height_frac)

    frames = [to_gray(frame0[:top_h, :]).astype(np.int16)]
    for idx in range(1, max_probe_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(to_gray(frame[:top_h, :]).astype(np.int16))
    cap.release()
    frames = np.asarray(frames)

    # calculate difference between adjacent frames per row, and set mask to True
    # if in the second half of the frames, the row is stable
    row_mask = []
    for r in range(frames.shape[1]):
        rows = frames[:, r, :]
        diffs = np.mean(np.abs(rows[:-1, :] - rows[1:, :]), axis=1)
        row_mask.append(diffs[diffs.shape[0] // 2 :].max() < mad_limit)
    row_mask = np.asarray(row_mask)
    best_start, best_end = find_longest_segment_in_mask(row_mask)
    if best_start is None:
        raise RuntimeError("Header not found")
    assert best_end is not None  # sanity check

    frames = frames[:, best_start : best_end + 1, :]
    diffs = np.mean(np.abs(frames[:-1, :, :] - frames[1:, :, :]), axis=(1, 2))
    start_idx = diffs.shape[0] - 1
    while start_idx > 1:
        if diffs[start_idx - 1] > mad_limit:
            break
        start_idx -= 1
    return start_idx, best_start, best_end


def detect_header_footer_bounds(
    video_path: Path,
    start_idx: int,
    header_end_row: int,
) -> int:
    """Detect the first row of the stable footer region below the header.

    The footer region is expected to have UI elements floating above content
    scrolling pass. Due to the UI elements, it is expected to have local regions
    that are stable across frames. Footer is located by detecting these regions.
    Only regions below the header region are considered.

    The detection is done by computing per-pixel standard deviation across frames
    starting at ``start_idx`` and identifies rows with notable areas of low
    temporal variance (stable footer content). It returns the top row index of
    the longest stable footer segment.

    Args:
        video_path: Path to the input video.
        start_idx: Frame index at which to begin analysis.
        header_end_row: Inclusive end row of the header region.

    Returns:
        The row index of the footer start in full-frame coordinates.

    Raises:
        RuntimeError: If a footer segment cannot be found.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
    frames = []
    if start_idx == 0:
        frames.append(to_gray(frame[header_end_row + 1 :, :]))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        if idx < start_idx:
            continue
        frames.append(to_gray(frame[header_end_row + 1 :, :]))
    frames = np.asarray(frames)
    std = np.std(frames, axis=0)
    row_mask = np.mean(std < 40, axis=1) > 0.1
    footer_start, _ = find_longest_segment_in_mask(row_mask)
    if footer_start is None:
        raise RuntimeError("Footer not found")
    return footer_start + header_end_row


def detect_carousel_segment(
    video_path: Path,
    start_idx: int,
    header_row_incl: tuple[int, int],
    footer_row: int,
    max_probe_frames: int = 80,
    upper_frac: float = 0.5,
    upper_spike_thresh: float = 12.0,
    lower_scroll_thresh: float = 6.0,
    scroll_consecutive: int = 3,
) -> tuple[tuple[int, int] | None, int | None, list[int]]:
    """Detect a carousel band and its frames before scrolling begins.

    This looks for a band in the upper half of the content area with transient
    spikes (image flips), while the lower half stays mostly stable. Scrolling is
    assumed to begin once the lower half shows sustained variance.

    Args:
        video_path: Path to the input video.
        start_idx: Frame index to begin probing.
        header_row_incl: Inclusive header row bounds.
        footer_row: Row index where the footer starts.
        max_probe_frames: Maximum number of frames to analyze.
        upper_frac: Fraction of the content height used as the upper band.
        upper_spike_thresh: Minimum spike magnitude in the upper band.
        lower_scroll_thresh: Threshold for sustained lower-band variance.
        scroll_consecutive: Frames of sustained lower activity to flag scroll start.

    Returns:
        A tuple of (carousel_row_incl, carousel_end_frame_idx, carousel_frames).
        ``carousel_row_incl`` and ``carousel_end_frame_idx`` are ``None`` when
        no carousel is detected. ``carousel_frames`` lists the frame indices of
        distinct carousel images.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None, None, []
    idx = 0
    while idx < start_idx:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None, None, []
        idx += 1

    content_start = header_row_incl[1] + 1
    content_end = footer_row
    if content_end <= content_start:
        cap.release()
        return None, None, []
    content_h = content_end - content_start
    upper_h = max(1, int(content_h * upper_frac))
    upper_slice = slice(0, upper_h)
    lower_slice = slice(upper_h, content_h)

    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    lower_activity: list[float] = []

    prev_gray = to_gray(frame).astype(np.int16)
    for _ in range(max_probe_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        cur_gray = to_gray(frame).astype(np.int16)
        diff = np.abs(cur_gray - prev_gray)
        mean_diff = float(np.mean(diff))
        if mean_diff < 1.0:
            prev_gray = cur_gray
            continue
        frames.append(cur_gray)
        frame_indices.append(idx)
        row_diff = np.mean(diff[content_start:content_end, :], axis=1)
        lower_activity.append(float(np.mean(row_diff[lower_slice])))
        prev_gray = cur_gray
    cap.release()

    if len(frames) < 2:
        return None, None, []

    stack = np.stack(frames, axis=0)[:, content_start:content_end, :]
    std = np.std(stack, axis=0)
    row_std = np.mean(std, axis=1)
    stable_mask = row_std < lower_scroll_thresh
    stable_start, stable_end = find_longest_segment_in_mask(stable_mask)
    if stable_start is None or stable_end is None:
        return None, None, []

    scroll_start_idx = None
    for i in range(0, len(lower_activity) - scroll_consecutive + 1):
        window = lower_activity[i : i + scroll_consecutive]
        if np.all(np.asarray(window) >= lower_scroll_thresh):
            scroll_start_idx = frame_indices[i]
            break

    if scroll_start_idx is not None:
        cut_idx = frame_indices.index(scroll_start_idx)
        frames = frames[:cut_idx]
        frame_indices = frame_indices[:cut_idx]
        stack = stack[:cut_idx, :, :]
        row_std = np.mean(np.std(stack, axis=0), axis=1)
    if not frames:
        return None, None, []

    upper_row_std = row_std[upper_slice]
    row_threshold = max(float(np.percentile(upper_row_std, 75)), upper_spike_thresh)
    row_mask = upper_row_std >= row_threshold
    band_start, band_end = find_longest_segment_in_mask(row_mask)
    if band_start is None or band_end is None:
        return None, None, []
    carousel_row_incl = (content_start + band_start, content_start + band_end)

    carousel_frames: list[int] = []
    carousel_frames.append(frame_indices[0])
    for i in range(1, len(frames)):
        cur = frames[i]
        prev = frames[i - 1]
        band_diff = np.mean(
            np.abs(cur - prev)[carousel_row_incl[0] : carousel_row_incl[1] + 1, :]
        )
        if band_diff >= upper_spike_thresh:
            carousel_frames.append(frame_indices[i])
    carousel_frames = sorted(set(carousel_frames))
    if not carousel_frames:
        return None, None, []

    carousel_end_idx = (
        scroll_start_idx if scroll_start_idx is not None else frame_indices[-1]
    )
    return carousel_row_incl, carousel_end_idx, carousel_frames


def auto_detect_layout(
    video_path: Path,
    header_max_probe_frames: int,
    header_top_probe_height_frac: float,
    header_mad_limit: float = 5,
    carousel_max_probe_frames: int = 80,
    carousel_upper_frac: float = 0.5,
    carousel_upper_spike_thresh: float = 12.0,
    carousel_lower_scroll_thresh: float = 6.0,
    carousel_scroll_consecutive: int = 3,
) -> Layout:
    """Automatically detect header/footer bounds and build a Layout object.

    Args:
        video_path: Path to the input video.
        header_max_probe_frames: Maximum number of frames to probe for header.
        header_top_probe_height_frac: Fraction of the height used for header probing.
        header_mad_limit: Mean absolute difference threshold for stability.

    Returns:
        A populated ``Layout`` with header/footer bounds and extracted header
        frame. ``Layout.header_frame`` has shape (H_header, W, 3) and dtype
        ``uint8``.

    Raises:
        RuntimeError: If the video cannot be read to extract header content.
    """
    start_idx, header_row_start, header_row_end = find_header_stable_start(
        video_path,
        max_probe_frames=header_max_probe_frames,
        top_probe_height_frac=header_top_probe_height_frac,
        mad_limit=header_mad_limit,
    )
    footer_row = detect_header_footer_bounds(
        video_path,
        start_idx=start_idx,
        header_end_row=header_row_end,
    )
    carousel_row_incl, carousel_end_frame_idx, carousel_frame_indices = (
        detect_carousel_segment(
            video_path,
            start_idx=start_idx,
            header_row_incl=(header_row_start, header_row_end),
            footer_row=footer_row,
            max_probe_frames=carousel_max_probe_frames,
            upper_frac=carousel_upper_frac,
            upper_spike_thresh=carousel_upper_spike_thresh,
            lower_scroll_thresh=carousel_lower_scroll_thresh,
            scroll_consecutive=carousel_scroll_consecutive,
        )
    )
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
    idx = 0
    while idx < start_idx:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Unable to read video for header content extraction")
        idx += 1
    return Layout(
        start_frame_idx=start_idx,
        header_row_incl=(header_row_start, header_row_end),
        footer_row=footer_row,
        header_frame=frame[header_row_start : header_row_end + 1, :, :],
        carousel_row_incl=carousel_row_incl,
        carousel_end_frame_idx=carousel_end_frame_idx,
        carousel_frame_indices=carousel_frame_indices,
    )
