#!/bin/bash

read -p "Paste HF token: " token
export HF_TOKEN="$token"
export MAX_JOBS=8
pip install -e .