#!/bin/bash
source ../venv/bin/activate
python3 -i test_all.py --dataset ~/Pictures/KNIVES_256_soft --img-size 32 \
    --arch-file configs/test_arch_custom.yaml \
    --training-file configs/test_updater.yaml \
    --experiment test_32_misc_001 \
    --storage-path ./OUTPUT \
