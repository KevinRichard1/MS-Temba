#!/usr/bin/env bash

set -e

export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH}

python MSTemba_main.py \
    -dataset ego4d_mq \
    -mode rgb \
    -backbone slowfast \
    -input_feat_dim 2304 \
    -num_classes 110 \
    -model mstemba \
    -train True \
    -rgb_root /data/asinha13/ego4d_data/v2/slowfast8x8_r101_k400 \
    -anno /data/asinha13/ego4d_data/v2/annotations/moments_train.json \
    -val_anno /data/asinha13/ego4d_data/v2/annotations/moments_val.json \
    -moment_classes /data/asinha13/ego4d_data/v2/annotations/ego4d_moment_classes.json \
    -num_clips 928 \
    -skip 0 \
    -comp_info False \
    -epochs 1 \
    -unisize True \
    -alpha_l 1 \
    -beta_l 0.05 \
    -batch_size 4 \
    --lr 5e-4 \
    --weight-decay 0.01 \
    -output_dir ./workdirs/