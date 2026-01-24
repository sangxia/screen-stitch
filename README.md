# screen-stitch

Stitch a vertically scrolling screen-recording video into a single tall image. The tool detects
the stable UI header/footer, isolates the scrolling content region, estimates per-frame scroll,
and concatenates newly revealed strips into a continuous image.

## Usage

```bash
uv run screen-stitch path/to/scroll.mp4 stitched.png
```

## Algorithm outline

1. **Probe the header**: Sample the top portion of early frames and compute per-row mean absolute
   differences to find a stable header band.
2. **Find the footer**: Analyze temporal variance below the header to locate stable UI elements
   in the footer region.
3. **Define the ROI**: Use the area between header and footer as the scrolling region.
4. **Estimate scroll**: For each frame, estimate vertical translation using phase correlation;
   validate with overlap NCC. If phase correlation fails, use template matching on the bottom strip
   to estimate scroll.
5. **Stitch**: Append the newly revealed strip each step and stack all strips with the original
   header.

## Module layout

- `screen_stitch/__init__.py`: CLI entry point, argument parsing and main logic.
- `screen_stitch/layout_detection.py`: Header/footer detection.
- `screen_stitch/stitch.py`: Scroll estimation and stitching logic.
- `screen_stitch/cvutils.py`: Grayscale conversion and correlation helpers.

## License

See [LICENSE](LICENSE).
