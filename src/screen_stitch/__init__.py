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

    # Auto-detection knobs (defaults intended to work broadly)
    ap.add_argument(
        "--max-probe-frames",
        type=int,
        default=600,
        help="Max frames scanned from start to find stable header start.",
    )
    ap.add_argument(
        "--top-probe-height",
        type=int,
        default=260,
        help="Top region height used to detect header stabilization.",
    )
    ap.add_argument(
        "--stable-window",
        type=int,
        default=15,
        help="Consecutive frames required under threshold to declare header stable.",
    )
    ap.add_argument(
        "--stable-mad-thresh",
        type=float,
        default=2.5,
        help="Mean absdiff threshold (0-255) in top probe to consider stable.",
    )
    ap.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Stride between sampled frames used for header/footer boundary detection.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=60,
        help="Max sampled frames used for boundary detection after stabilization.",
    )
    ap.add_argument(
        "--header-row-diff-thresh",
        type=float,
        default=4.5,
        help="Row mean absdiff threshold (0-255) to classify header rows as stable.",
    )
    ap.add_argument(
        "--footer-row-diff-thresh",
        type=float,
        default=6.0,
        help="Row mean absdiff threshold (0-255) to classify footer rows as stable.",
    )
    ap.add_argument(
        "--min-footer-height",
        type=int,
        default=50,
        help="Minimum footer height (px) to accept a detected bottom stable band.",
    )
    ap.add_argument(
        "--boundary-smooth-k",
        type=int,
        default=9,
        help="Smoothing window (rows) for boundary detection signals.",
    )
    ap.add_argument(
        "--detect-max-dim",
        type=int,
        default=900,
        help="Downscale max dimension for detection computations.",
    )

    # Stitching knobs
    ap.add_argument(
        "--min-scroll-px", type=int, default=3, help="Minimum scroll to append (px)."
    )
    ap.add_argument(
        "--min-overlap-ncc",
        type=float,
        default=0.65,
        help="Min NCC for overlap validation.",
    )
    ap.add_argument(
        "--min-phase-response",
        type=float,
        default=0.10,
        help="Min phaseCorrelate response to trust dy.",
    )
    ap.add_argument(
        "--tmpl-h",
        type=int,
        default=200,
        help="Template height (px) for fallback template matching.",
    )
    ap.add_argument(
        "--min-tmpl-score",
        type=float,
        default=0.80,
        help="Min template match score to accept fallback.",
    )
    ap.add_argument(
        "--max-dx-allowed",
        type=float,
        default=4.0,
        help="Reject if |dx| exceeds this (px).",
    )
    ap.add_argument(
        "--max-dim",
        type=int,
        default=900,
        help="Downscale max dimension for stitching computations.",
    )

    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()

    layout = auto_detect_layout(
        args.video_file,
        header_max_probe_frames=200,
        header_top_probe_height_frac=0.3,
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
            phase_max_dx_allowed_px=4,
            phase_min_response=0.1,
            min_scroll_frac=0.3,
            template_min_score=0.8,
            phase_min_overlap_ncc=0.65,
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
