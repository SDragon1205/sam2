#!/bin/bash

# for f in tt20_*; do
#   mv "$f" "${f/tt20_/tt21_}"
# done
for f in tt20_*; do
  echo mv "$f" "${f/tt20_/tt21_}"
done