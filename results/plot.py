import pylab

import numpy as np
import pylab as pl
from scipy import interpolate, signal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import csv

# We need a special font for the code below.  It can be downloaded this way:
import os
import urllib2
if not os.path.exists('Humor-Sans.ttf'):
    fhandle = urllib2.urlopen('http://antiyawn.com/uploads/Humor-Sans-1.0.ttf')
    open('Humor-Sans.ttf', 'wb').write(fhandle.read())

plt.xkcd()

# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter=',')  # create reader
        headers = reader.next()  # skip header
        for row in reader:
            item = row
            data.append(item)
    f.close()
    return data
    
def makePlot(xAxis, yAxes, lineLabels, xLabel, yLabel, title) :
    
    ax = pylab.axes()

    for line in range(0, len(lineLabels)) :
        ax.plot(xAxis, yAxes[line], lw=1, label=lineLabels[line], marker='o')
        
    #ax.plot(data[:3,1], data[:3,2], 'b', lw=1, label='MAP', marker='o')
    #ax.plot(data[:3,1], data[:3,3], 'r', lw=1, label='MAR', marker='o')
    #ax.plot(data[:3,1], data[:3,4], 'g', lw=1, label='FSCORE', marker='o')

    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.legend(loc='best')
    
    ax.set_xlim(float(min(xAxis)) -0.5, float(max(xAxis)) + 0.5)
    ax.set_ylim(float(yAxes.min() -0.5), float(yAxes.max()) +0.5 )
    
    plt.savefig(title + ".png")
    plt.clf()

if __name__ == "__main__":
    np.random.seed(0)
    
    data = np.genfromtxt("RB.csv", delimiter=',', skip_header=1)
    
    lineLabels = ["MAP","MAR","FSCORE"]
    
    yAxes = data[:3,2:5]
    yAxes = np.transpose(yAxes)
    
    makePlot(data[:3,1], yAxes, lineLabels, "Artists", "Score", "Random baseline recommender for different Artist numbers")
    
    data = np.genfromtxt("CF.csv", delimiter=',', skip_header=1)
    
    yAxes = data[:3,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[:3,1], yAxes, lineLabels, "Artists", "Score", "CF recommender for different Artist numbers with " + str(int(data[0,0])) + " neighbours")
    
    yAxes = data[3:6,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[3:6,1], yAxes, lineLabels, "Artists", "Score", "CF recommender for different Artist numbers with " + str(int(data[3,0])) + " neighbours")
    
    yAxes = data[6:9,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[6:9,1], yAxes, lineLabels, "Artists", "Score", "CF recommender for different Artist numbers with " + str(int(data[6,0])) + " neighbours")
    
    data = np.genfromtxt("CB.csv", delimiter=',', skip_header=1)
    
    yAxes = data[:3,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[:3,1], yAxes, lineLabels, "Artists", "Score", "CB recommender for different Artist numbers with " + str(int(data[0,0])) + " neighbours")
    
    yAxes = data[3:6,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[3:6,1], yAxes, lineLabels, "Artists", "Score", "CB recommender for different Artist numbers with " + str(int(data[3,0])) + " neighbours")
    
    yAxes = data[6:9,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[6:9,1], yAxes, lineLabels, "Artists", "Score", "CB recommender for different Artist numbers with " + str(int(data[6,0])) + " neighbours")
    
    data = np.genfromtxt("Hybrid.csv", delimiter=',', skip_header=1)
    
    yAxes = data[:3,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[:3,1], yAxes, lineLabels, "Artists", "Score", "Hybrid recommender for different Artist numbers with " + str(int(data[0,0])) + " neighbours")
    
    yAxes = data[3:6,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[3:6,1], yAxes, lineLabels, "Artists", "Score", "Hybrid recommender for different Artist numbers with " + str(int(data[3,0])) + " neighbours")
    
    yAxes = data[6:9,2:5]
    yAxes = np.transpose(yAxes)    
    makePlot(data[6:9,1], yAxes, lineLabels, "Artists", "Score", "Hybrid recommender for different Artist numbers with " + str(int(data[6,0])) + " neighbours")
    
    