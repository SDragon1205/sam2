#!/bin/bash


export SCALE=s
export BATCH_SIZE=128
python3 train/v2v_SAVPE/Object_Oriented/DINOv2/v11_SAVPE_DINOv2_L_multi_layer_135_OO.py
export SCALE=m
export BATCH_SIZE=128
python3 train/v2v_SAVPE/Object_Oriented/DINOv2/v11_SAVPE_DINOv2_L_multi_layer_135_OO.py
export SCALE=l
export BATCH_SIZE=64
python3 train/v2v_SAVPE/Object_Oriented/DINOv2/v11_SAVPE_DINOv2_L_multi_layer_135_OO.py
