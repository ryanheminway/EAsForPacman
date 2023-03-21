#!/bin/bash
# SCRIPT FOR LAUNCHING SLURM JOBS FROM ANOTHER SCRIPT
cd /home/heminway.r/EAsForPacman-main/
source activate EAResearch
python pacman.py -k 1 -g DirectionalGhost -l smallClassicGA --gens 1000 --pop 1000 --mutation $1 --crossover $2 --genetic 