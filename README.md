# screen-stitch

Stitch a vertically scrolling screen-recording video into a single tall PNG. The tool detects the stable UI header/footer, isolates the scrolling content region, estimates per-frame scroll, and concatenates newly revealed strips into a continuous image.

## Features

- **Automatic layout detection** of stable header/footer regions in screen recordings.
- **Robust scroll estimation** using phase correlation with template-matching fallback.
- **Single-file output** (PNG) suitable for documentation or archival.
- **CLI-first workflow** with tunable parameters for different UIs.

## Installation

```bash
python -m pip install -e .
```

## Usage

```bash
screen-stitch path/to/scroll.mp4 stitched.png
```

### Common options

```bash
screen-stitch path/to/scroll.mp4 stitched.png \
  --header-max-probe-frames 300 \
  --header-top-probe-height-frac 0.25 \
  --header-mad-limit 4.0 \
  --min-scroll-frac 0.3 \
  --phase-max-dx-allowed-px 4 \
  --phase-min-response 0.1 \
  --phase-min-overlap-ncc 0.65 \
  --template-min-score 0.8
```

## How it works (algorithm outline)

1. **Probe the header**: Sample the top portion of early frames and compute per-row mean absolute differences to find a stable header band.
2. **Find the footer**: Analyze temporal variance below the header to locate stable UI elements in the footer region.
3. **Define the ROI**: Use the area between header and footer as the scrolling region.
4. **Estimate scroll**: For each frame, estimate vertical translation using phase correlation; validate with overlap NCC.
5. **Fallback matching**: If phase correlation fails, use template matching on the bottom strip to estimate scroll.
6. **Stitch**: Append the newly revealed strip each step and stack all strips with the original header.

## Module layout

- `screen_stitch/__init__.py`: CLI entry point and argument parsing.
- `screen_stitch/layout_detection.py`: Header/footer detection and layout assembly.
- `screen_stitch/stitch.py`: Scroll estimation and stitching logic.
- `screen_stitch/cvutils.py`: Grayscale conversion and correlation helpers.

## Output

The tool outputs a single PNG image containing the stitched vertical scroll. The filename is taken from the positional `output` argument.

## Requirements

- Python 3.11+
- OpenCV (`opencv-contrib-python`)
- NumPy

Dependencies are declared in `pyproject.toml`.

## Troubleshooting

- **Header not found**: Increase `--header-max-probe-frames`, adjust `--header-top-probe-height-frac`, or relax `--header-mad-limit`.
- **Footer not found**: Ensure the recording includes a stable footer area; try shorter clips with clear UI chrome.
- **Bad stitches**: Increase `--phase-min-overlap-ncc` for stricter matching, or adjust `--min-scroll-frac` for faster/slower scrolls.

## Development

Run the CLI directly:

```bash
python -m screen_stitch path/to/scroll.mp4 stitched.png
```

## License

See [LICENSE](LICENSE).
