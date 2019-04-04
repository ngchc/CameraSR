#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./results \
    --summary_dir ./results/log \
    --mode inference \
    --is_training False \
	--task SRGAN \
    --input_dir_LR ./testset \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./model/model
