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


def find_longest_segment_in_mask(mask: np.ndarray) -> tuple[int | None, int | None]:
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
    """
    Returns:
        the first frame to cut the video from, the row start and row end (inclusive)
        of the header area
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
    """ """
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


def auto_detect_layout(
    video_path: Path,
    header_max_probe_frames: int,
    header_top_probe_height_frac: float,
    header_mad_limit: float = 5,
) -> Layout:
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
    )
