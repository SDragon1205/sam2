#!/bin/bash

filename=$(basename "$1" .py)

# Create temporary job script
cat > temp_job.sh << EOL
#!/bin/bash
#SBATCH --account PES0812
#SBATCH --job-name=${filename}
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_log/${filename}_output.log
#SBATCH --error=slurm_log/${filename}_error.log
#SBATCH --mem=256000

cd $HOME/Documents/v2vdet
module purge
module load pytorch/2.5.0
module load hpcx/2.17.1
source ~/.bashrc
conda env config vars set PYTHONPATH=. --name v2vdet
conda activate v2vdet
yolo settings datasets_dir="DATASET"
yolo settings weights_dir="ckpt"
yolo settings wandb=True
yolo settings runs_dir="v2v_training_result"
python $1
EOL

sbatch temp_job.sh

rm temp_job.sh