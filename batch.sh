#!/bin/bash

for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    uv run screen-stitch "$@"
    exit 0
  fi
done

for filename in ./data/*; do
  echo "$filename"
  uv run screen-stitch "$filename" "out/$(date +%s).png" "$@"
done
