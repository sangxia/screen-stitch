#!/bin/bash

for filename in ./data/*; do
  echo $filename
  uv run screen-stitch "$filename" out/$(date +%s).png --min-scroll-frac 0.2
done
