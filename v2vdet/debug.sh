#!/bin/bash

if [ -n "$2" ]; then
    export CUDA_VISIBLE_DEVICES=$2
else
    unset CUDA_VISIBLE_DEVICES
fi

if [ -n "$3" ]; then
    export DEBUGPY_PORT=$3
else
    export DEBUGPY_PORT=4998
fi

debugpy --listen 0.0.0.0:${DEBUGPY_PORT} --wait-for-client $1

unset CUDA_VISIBLE_DEVICES
unset DEBUGPY_PORT
