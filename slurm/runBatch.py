#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=RyanEAPacman
#SBATCH --mem=20G
#SBATCH --partition=short
#SBATCH --output=batchLog.txt
cd /home/heminway.r/EAsForPacman-main/
source activate EAResearch
python pacman.py -k 1 -g DirectionalGhost -l smallClassicGA --genetic 
