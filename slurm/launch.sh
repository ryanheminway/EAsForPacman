#!/bin/bash
# SCRIPT FOR LAUNCHING SLURM JOBS FROM ANOTHER SCRIPT
cd /home/heminway.r/EAsForPacman-main/
source activate EAResearch
python pacman.py -k 0 -g DirectionalGhost -l smallClassicGA --gens 10000 --pop 100 --mutation $1 --crossover $2 --tournySize $3 --genetic 