#!/bin/bash
python train_moa.py \
    --img_root_path /path/to/your/generated/bbbc021/images \
    --config_path ../configs/bbbc021_all.yaml \
    --mode eval \
    --ckpt_path checkpoint.pth \
    