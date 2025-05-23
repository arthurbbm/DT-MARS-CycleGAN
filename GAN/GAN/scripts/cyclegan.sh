#!/bin/bash

accelerate launch --multi_gpu --num_processes 10 --num_machines 1 --gpu_ids 0,1,2,3,4,5,6,7,8,9 train_cycleGAN_acc.py \
    --batchSize 8 \
    --n_epochs 2 \
    --outdir 'output/test' \
    --decay_epoch 1