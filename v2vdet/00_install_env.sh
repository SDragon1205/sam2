#!/bin/bash

conda install mamba
mamba install pytorch torchvision torchaudio transformers ultralytics supervision python-lmdb wandb pycocotools -c conda-forge