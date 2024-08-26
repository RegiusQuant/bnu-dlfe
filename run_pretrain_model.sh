#!/bin/bash
# python run_pretrain_model.py \
# --input_path data\pretrain\pretrain.csv \
# --scaler_path outputs\scaler\scaler.pkl \
# --group_cols C_1 \
# --mask_prob 0.2 \
# --num_feat 32

python run_pretrain_model.py \
--input_path data/pretrain.csv \
--scaler_path outputs/scaler/scaler.pkl \
--mask_prob 0.2 \
--num_feat 46 \
--num_hidden 128 \
--max_epochs 3

