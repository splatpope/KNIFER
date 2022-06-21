#!/bin/bash
source ../venv/bin/activate
python3 -i test_all.py --dataset ~/Pictures/KNIVES_256_soft --img-size 64 --arch-file configs/test_arch.yaml --training-file configs/test_updater.yaml --experiment test_64_attn --storage-path ./OUTPUT
