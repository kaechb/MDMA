#!/bin/bash
#SBATCH --partition=allgpu,maxgpu
#SBATCH --constraint='A100'|'P100'|'V100'|
#SBATCH --time=72:00:00                           # Maximum time requested
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --chdir=/home/kaechben/slurm_thesis        # directory must already exist!
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=NONE                       # Type of email notification- BEGIN,END,FAIL,ALL

export WANDB__SERVICE_WAIT=300
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate mdma-cfm
cd /home/$USER/MDMACalo/
python3 main.py jet_nf
