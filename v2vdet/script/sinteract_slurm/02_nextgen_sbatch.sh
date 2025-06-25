#!/bin/bash
#SBATCH -A PES0812
#SBATCH -c 8
#SBATCH -p nextgen
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=256000
#SBATCH -t 24:00:00
#SBATCH --job-name=v2vdet
#SBATCH -o v2vdet_output.log
#SBATCH -e v2vdet_error.log

source ~/.bashrc
conda activate v2vdet 

if [ -z "$1" ]; then
    echo "Error: Please provide a Python script to execute!" >&2
    echo "Usage: sbatch job_script.sh <python_script.py>" >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: The specified Python script '$1' does not exist!" >&2
    exit 1
fi	
	
python3 "$1"
