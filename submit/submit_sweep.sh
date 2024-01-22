#!/bin/bash
#SBATCH --partition=allgpu,maxgpu
#SBATCH --constraint='A100'|'P100'|'V100'|
#SBATCH --time=72:00:00                           # Maximum time requested
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --chdir=/home/kaechben/slurm_calo_middle        # directory must already exist!
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=NONE                       # Type of email notification- BEGIN,END,FAIL,ALL

export WANDB_PROJECT="MDMA_jet_fm"
export WANDB_ENTITY="kaechben"
export WANDB__SERVICE_WAIT=300
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate mdma-cfm
cd /home/$USER/MDMACalo/
wandb login f39ea2cc30c7a621000b7fa3355a8c0e848a91d3
wandb agent --count 1 11x6nhtd

