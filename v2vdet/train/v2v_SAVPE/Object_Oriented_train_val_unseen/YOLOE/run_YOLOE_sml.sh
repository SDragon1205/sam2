#!/bin/bash

for SCALE in s m l
do
  if [[ "$SCALE" == "s" || "$SCALE" == "m" ]]; then
    export BATCH_SIZE=64
  else
    export BATCH_SIZE=32
  fi
  python3 train/v2v_SAVPE/Object_Oriented/YOLOE/v11_SAVPE_YOLOE_OO.py
done
