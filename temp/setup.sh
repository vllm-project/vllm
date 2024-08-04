#!/bin/bash

read -p "Paste HF token: " token
export HF_TOKEN="$token"
export MAX_JOBS=8
git config --global user.email "felixzhu555@gmail.com"
git config --global user.name "Felix Zhu"
pip install -e .