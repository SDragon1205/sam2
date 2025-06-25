#!/bin/bash

RUNNER="sh script/sbatch_slurm/02_sbatch_2GPU_40G_A100.sh"

FILES=(
    "train/exp_different_template_backbone/CLIP/CLIP_FT.py"
    "train/exp_different_template_backbone/SigLIP/SigLIP.py"
    "train/exp_different_template_backbone/SigLIP/SigLIP_FT.py"
    "train/exp_different_template_backbone/DINO2/DINO2_FT.py"
    "train/exp_different_template_backbone/DINO2_with_reg/DINO2r_FT.py"
    "train/exp_different_template_backbone/SigLIP2/SigLIP2_FT.py"
)

for file in "${FILES[@]}"; do
    $RUNNER $BASE_DIR/$file
done

# FILES=(
#     "CLIP_FT.py"
#     "CLIP_FT_multi_layer_246.py"
#     "CLIP_FT_multi_layer_2468.py"
#     "CLIP_multi_layer_246.py"
#     "CLIP_multi_layer_2468.py"
# )

# for file in "${FILES[@]}"; do
#     $RUNNER $BASE_DIR/$file
# done

# BASE_DIR="train/exp_different_template_backbone/SigLIP"
# FILES=(
#     "SigLIP.py"
#     "SigLIP_FT.py"
#     "SigLIP_FT_multi_layer_246.py"
#     "SigLIP_FT_multi_layer_2468.py"
#     "SigLIP_multi_layer_246.py"
#     "SigLIP_multi_layer_2468.py"
# )

# for file in "${FILES[@]}"; do
#     $RUNNER $BASE_DIR/$file
# done

# BASE_DIR="train/exp_different_template_backbone/SigLIP2"
# FILES=(
#     "SigLIP2.py"
#     "SigLIP2_FT.py"
#     "SigLIP2_FT_multi_layer_246.py"
#     "SigLIP2_FT_multi_layer_2468.py"
#     "SigLIP2_multi_layer_246.py"
#     "SigLIP2_multi_layer_2468.py"
# )

# for file in "${FILES[@]}"; do
#     $RUNNER $BASE_DIR/$file
# done

# BASE_DIR="train/exp_different_template_backbone/DINO2"
# FILES=(
#     "DINO2.py"
#     "DINO2_FT.py"
#     "DINO2_FT_multi_layer_246.py"
#     "DINO2_FT_multi_layer_2468.py"
#     "DINO2_multi_layer_246.py"
#     "DINO2_multi_layer_2468.py"
# )

# for file in "${FILES[@]}"; do
#     $RUNNER $BASE_DIR/$file
# done

# BASE_DIR="train/exp_different_template_backbone/DINO2_with_reg"
# FILES=(
#     "DINO2r.py"
#     "DINO2r_FT.py"
#     "DINO2r_FT_multi_layer_246.py"
#     "DINO2r_FT_multi_layer_2468.py"
#     "DINO2r_multi_layer_246.py"
#     "DINO2r_multi_layer_2468.py"
# )

# for file in "${FILES[@]}"; do
#     $RUNNER $BASE_DIR/$file
# done