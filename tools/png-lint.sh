#!/bin/bash

# Ensure that *.excalidraw.png files have the excalidraw metadata
# embedded in them. This ensures they can be loaded back into
# the tool and edited in the future.

find . -iname '*.excalidraw.png' | while read -r file; do
	if git check-ignore -q "$file"; then
		continue
	fi
	if ! grep -q "excalidraw+json" "$file"; then
		echo "$file was not exported from excalidraw with 'Embed Scene' enabled."
		exit 1
	fi
done
