import argparse
from screen_stitch.layout_detection import auto_detect_layout
from screen_stitch.stitch import stitch_video, StitchParams
from pathlib import Path
import cv2


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stitch a vertically scrolling screen-recording video into one tall PNG."
    )
    ap.add_argument("video_file", help="Input video file.", type=Path)
    ap.add_argument("output", help="Output image file name.", type=Path)

    # Auto-detection knobs
    ap.add_argument(
        "--header-max-probe-frames",
        type=int,
        default=200,
        help="Max frames scanned from start to find stable header start.",
    )
    ap.add_argument(
        "--header-top-probe-height-frac",
        type=float,
        default=0.3,
        help="Top probe height as a fraction of frame height.",
    )
    ap.add_argument(
        "--header-mad-limit",
        type=float,
        default=5.0,
        help="Mean absdiff threshold (0-255) for header stabilization.",
    )

    # Stitching knobs
    ap.add_argument(
        "--phase-max-dx-allowed-px",
        type=int,
        default=4,
        help="Reject if |dx| exceeds this (px).",
    )
    ap.add_argument(
        "--phase-min-response",
        type=float,
        default=0.1,
        help="Min phaseCorrelate response to trust dy.",
    )
    ap.add_argument(
        "--min-scroll-frac",
        type=float,
        default=0.3,
        help="Minimum scroll as a fraction of ROI height.",
    )
    ap.add_argument(
        "--template-min-score",
        type=float,
        default=0.8,
        help="Min template match score to accept fallback.",
    )
    ap.add_argument(
        "--phase-min-overlap-ncc",
        type=float,
        default=0.65,
        help="Min NCC for overlap validation.",
    )

    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()

    layout = auto_detect_layout(
        args.video_file,
        header_max_probe_frames=args.header_max_probe_frames,
        header_top_probe_height_frac=args.header_top_probe_height_frac,
        header_mad_limit=args.header_mad_limit,
    )
    print(
        (
            f"Found start frame {layout.start_frame_idx}, "
            f"header row range {layout.header_row_incl}, "
            f"footer row {layout.footer_row}"
        )
    )
    img = stitch_video(
        args.video_file,
        layout,
        StitchParams(
            phase_max_dx_allowed_px=args.phase_max_dx_allowed_px,
            phase_min_response=args.phase_min_response,
            min_scroll_frac=args.min_scroll_frac,
            template_min_score=args.template_min_score,
            phase_min_overlap_ncc=args.phase_min_overlap_ncc,
        ),
    )
    cv2.imwrite(str(args.output), img)

    # layout, stable_frame = auto_detect_layout(args)

    # sp = StitchParams(
    #     video_file=args.video_file,
    #     out_png=args.output_png,
    #     start_frame_idx=layout.start_frame_idx,
    #     header_y1=layout.header_y1,
    #     footer_y0=layout.footer_y0,
    #     min_scroll_px=args.min_scroll_px,
    #     min_overlap_ncc=args.min_overlap_ncc,
    #     min_phase_response=args.min_phase_response,
    #     tmpl_h=args.tmpl_h,
    #     min_tmpl_score=args.min_tmpl_score,
    #     max_dx_allowed=args.max_dx_allowed,
    #     max_dim=args.max_dim,
    #     verbose=args.verbose,
    # )

    # stitch_video(sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
