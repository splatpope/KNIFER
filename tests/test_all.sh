#!/bin/bash
source ../venv/bin/activate
DSET=~/Pictures/KNIVES_256_soft
ISIZE=32

EXP="$1"
if [ -z $EXP ]
then
    echo "Missing experiment name.";
    exit 1;
fi

python3 -i test_all.py --dataset "$DSET" --img-size "$ISIZE" \
    --arch-file configs/test_arch_custom.yaml \
    --training-file configs/test_updater.yaml \
    --experiment "$EXP" \
    --storage-path ./OUTPUT \
