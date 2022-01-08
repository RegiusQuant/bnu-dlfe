#!/bin/bash

python run_feature_generation.py \
--input_path data\task\train.csv \
--scaler_path outputs\scaler\scaler.pkl \
--model_path outputs\model.ckpt \
--save_path outputs\train.npy
