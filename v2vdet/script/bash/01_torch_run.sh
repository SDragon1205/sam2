#!/bin/bash

# export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

torchrun --nproc_per_node 8 --master_port $2 $1 | tee output.log