#!/bin/bash
module purge
module load 2025
module load Anaconda3/2025.06-1
source activate vaes
python test_gyr.py
