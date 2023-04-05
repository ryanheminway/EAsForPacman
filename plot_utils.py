# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 08:57:30 2023

@author: hemin
"""
import os
import matplotlib.pyplot as plt
from pathlib import Path

def plot_scores(file_path):
    """
    Given a file path, this function will make a line graph for all of the 
    *.log files in that directory. It assumes a basic format for the log file
    where each line reports a single generation's fitness after a ":". 
    """
    files = [f for f in os.listdir(file_path) if f.endswith('.LOG')]
    fig, ax = plt.subplots()
    for file in files:
        with open(os.path.join(file_path, file), 'r') as f:
            scores = [float(line.split(':')[-1]) for line in f]
        if (scores[-1] < 700):   
            ax.plot(scores, label=file)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Model Performance')
    ax.set_title('GA Pacman Results')
    ax.legend()
    plt.show()
    plt.legend(loc='best', fancybox=True, shadow=True)

def find_bad_logs(file_path):
    """
    Given a directory file path, return a list of all file names which correspond
    to logs with errors or bad formatting. Really anything that will break the
    `plot_scores` function.
    """
    badfiles = list()
    files = [f for f in os.listdir(file_path) if f.endswith('.LOG')]
    fig, ax = plt.subplots()
    for file in files:
        with open(os.path.join(file_path, file), 'r') as f:
            try:
                scores = [float(line.split(':')[-1]) for line in f]
            except: 
                badfiles.append(file)
                
    return badfiles
    


# -------------- TEST BED ----------------- #        
if __name__ == '__main__':
    logDir = str(Path.cwd()) + "/models/" + "zeroGhostsAnnealing20230404/"
    #print(find_bad_logs(logDir))
    plot_scores(logDir)