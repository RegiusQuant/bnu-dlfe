#!/bin/bash

python run_few_shot_prediction.py \
--train_path data/train.csv \
--test_path data/test.csv \
--scaler_path outputs/scaler/scaler.pkl \
--model_path outputs/model.ckpt
