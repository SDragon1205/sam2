#!/bin/bash
conda env config vars set PYTHONPATH=.:perception_models
ln -s /DATA3/DATASET .
yolo settings datasets_dir="DATASET"
yolo settings weights_dir="ckpt"
yolo settings wandb=True
yolo settings runs_dir="v2v_training_result"