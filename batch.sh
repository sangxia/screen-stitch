while IFS= read -r line; do echo $line; uv run screen-stitch $line out/$(date +%s).png; done < <(ls data/*)
