#!/bin/bash
pip uninstall torch

# Initialize Conda for the bash shell
conda init bash

# Activate the desired Conda environment
conda create -n myenv python=3.8 -y

conda run -n myenv pip install debugpy

conda run -n myenv pip install supabase

conda run -n myenv pip install python-dotenv
