#!/bin/bash

for filename in ./data/*; do
  echo $filename
  uv run screen-stitch "$filename" out/$(date +%s).png
done
