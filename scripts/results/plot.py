import pylab

import numpy as np
import pylab as pl
from scipy import interpolate, signal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import csv
import sys

from textwrap import wrap

# We need a special font for the code below.  It can be downloaded this way:
import os
import urllib2


# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=',')  # create reader
        #headers = reader.next()  # skip header
        for row in reader:
            item = row
            data.append(item)
    f.close()
    return data
    
def makePlot(xAxes, yAxes, lineLabels, xLabel, yLabel, title) :
    
    ax = pylab.axes()

    for line in range(0, len(lineLabels)) :
        ax.plot(xAxes[line], yAxes[line], lw=1, label=lineLabels[line], marker='o')

   
    tempTitle = title
    title = ax.set_title("\n".join(wrap(title)))

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.legend(loc='best')
    
    #ax.set_xlim(float(xAxes.min() -0.5), float(xAxes.max()) + 0.5)
    #ax.set_ylim(float(yAxes.min() -0.5), float(yAxes.max()) + 0.5)
    
    plt.savefig(tempTitle + ".png")
    plt.clf()

if __name__ == "__main__":
    np.random.seed(0)
    
    fileName = "CF_k.csv"
    data = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    fileName = "UB_k.csv"
    data2 = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    fileName = "RB_k.csv"
    data3 = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    fileName = "RBU_k.csv"
    data4 = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    fileName = "CB_k.csv"
    data5 = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    fileName = "PB_k.csv"
    data6 = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    fileName = "UBCF_k.csv"
    data7 = np.genfromtxt(fileName, delimiter='\t', skip_header=0)
    
    lineLabels = ["CF", "UB", "RB", "RBU", "CB", "PB", "UBCF"]
    
    yAxes = [data[:,2], data2[:,2], data3[:,2], data4[:,2], data5[:,2], data6[:,2], data7[:,2]]
    xAxes = [data[:,3], data2[:,3], data3[:,3], data4[:,3], data5[:,3], data6[:,3], data7[:,3]]
    
    makePlot(xAxes, yAxes, lineLabels, "Mean Average Recall (%)", "Mean Average Precision (%)", "Precision Recall Plot of various Recommenders (recommended Artists from 10-40 in steps of five)")
    
    xAxes = [data[:,1], data[:,1], data[:,1], data[:,1], data[:,1], data[:,1], data[:,1]]
    yAxes = [data[:,4], data2[:,4], data3[:,4], data4[:,4], data5[:,4], data6[:,4], data7[:,4]]
    
    makePlot(xAxes, yAxes, lineLabels, "Number of Recommended Artists", "F1Score", "F1Score over number of artists recommended")