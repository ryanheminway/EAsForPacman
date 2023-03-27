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
        ax.plot(scores, label=file)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Model Performance')
    ax.set_title('GA Pacman Results')
    ax.legend()
    plt.show()



# -------------- TEST BED ----------------- #        
if __name__ == '__main__':
    logDir = str(Path.cwd()) + "/models/" + "testMoveLimits20230326/"
    plot_scores(logDir)