#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=RyanEAPacman
#SBATCH --mem=20G
#SBATCH --partition=short
#SBATCH --output=batchLog.txt
cd /home/heminway.r/EAsForPacman-main/
source activate EAResearch
python pacman.py -k 1 -g DirectionalGhost -l smallClassicGA --gens 1000 --pop 1000 --mutation 0.05 --crossover 0.5 --genetic 
