#!/bin/bash

for NUM_CLASSES in 600 800 1000 1200
do
  export NUM_CLASSES=$NUM_CLASSES
  python3 v2vdet_ultralytics/utils/ABO_DATASET_Create/main.py
done