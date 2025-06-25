#!/bin/bash

for SCALE in n s m l x
do
  export SCALE=$SCALE
  python3 train/v2v_SAVPE/Object_Oriented/v11_SAVPE_PE_L14_FT_multi_layer_678_Object365_oo.py
done

for SCALE in s m
do
  export SCALE=$SCALE
  python3 train/v2v_SAVPE/Object_Oriented/v11_SAVPE_PE_L14_multi_layer_678_Object365_oo.py
done