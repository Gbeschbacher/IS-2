import pylab

import numpy as np
import pylab as pl
from scipy import interpolate, signal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import csv

from textwrap import wrap

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
    
def makePlot(xAxes, yAxes, lineLabels, xLabel, yLabel, title) :
    
    ax = pylab.axes()

    for line in range(0, len(lineLabels)) :
        ax.plot(xAxes[line], yAxes[line], lw=1, label=lineLabels[line], marker='o')

   
    tempTitle = title
    #fig = plt.figure()
    #ax.set_title(title)
    title = ax.set_title("\n".join(wrap(title)))

    #fig.tight_layout()
    #title.set_y(1.05)
    #fig.subplots_adjust(top=0.8)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.legend(loc='best')
    
    ax.set_xlim(float(xAxes.min() -0.5), float(xAxes.max()) + 0.5)
    ax.set_ylim(float(yAxes.min() -0.5), float(yAxes.max()) + 0.5)
    
    plt.savefig(tempTitle + ".png")
    plt.clf()

if __name__ == "__main__":
    np.random.seed(0)
    
    data = np.genfromtxt("RB.csv", delimiter=',', skip_header=1)
    
    lineLabels = ["MAP","MAR","FSCORE"]
    
    yAxes = data[:3,2:5]
    yAxes = np.transpose(yAxes)
    
    xAxes = [data[:3,1], data[:3,1], data[:3,1]]
    xAxes = np.array(xAxes)
    print xAxes
    
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "Random baseline recommender for different Artist numbers")
    
    dataCF = np.genfromtxt("CF.csv", delimiter=',', skip_header=1)
    
    
    yAxes = dataCF[:3,2:5]
    yAxes = np.transpose(yAxes)    
    xAxes = [dataCF[:3,1], dataCF[:3,1], dataCF[:3,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "CF recommender for different Artist numbers with " + str(int(dataCF[0,0])) + " neighbours")
    
    yAxes = dataCF[3:6,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataCF[3:6,1], dataCF[3:6,1], dataCF[3:6,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "CF recommender for different Artist numbers with " + str(int(dataCF[3,0])) + " neighbours")
    
    yAxes = dataCF[6:9,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataCF[6:9,1], dataCF[6:9,1], dataCF[6:9,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "CF recommender for different Artist numbers with " + str(int(dataCF[6,0])) + " neighbours")
    
    dataCB = np.genfromtxt("CB.csv", delimiter=',', skip_header=1)
    
    yAxes = dataCB[:3,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataCB[:3,1], dataCF[:3,1], dataCF[:3,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "CB recommender for different Artist numbers with " + str(int(dataCB[0,0])) + " neighbours")
    
    yAxes = dataCB[3:6,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataCB[3:6,1], dataCF[3:6,1], dataCF[3:6,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "CB recommender for different Artist numbers with " + str(int(dataCB[3,0])) + " neighbours")
    
    yAxes = dataCB[6:9,2:5]
    yAxes = np.transpose(yAxes)   
    xAxes = [dataCB[6:9,1], dataCF[6:9,1], dataCF[6:9,1]]
    xAxes = np.array(xAxes)    
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "CB recommender for different Artist numbers with " + str(int(dataCB[6,0])) + " neighbours")
    
    dataHybrid = np.genfromtxt("Hybrid.csv", delimiter=',', skip_header=1)
    
    yAxes = dataHybrid[:3,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataHybrid[:3,1], dataCF[:3,1], dataCF[:3,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "Hybrid recommender for different Artist numbers with " + str(int(dataHybrid[0,0])) + " neighbours")
    
    yAxes = dataHybrid[3:6,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataHybrid[3:6,1], dataCF[3:6,1], dataCF[3:6,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "Hybrid recommender for different Artist numbers with " + str(int(dataHybrid[3,0])) + " neighbours")
    
    yAxes = dataHybrid[6:9,2:5]
    yAxes = np.transpose(yAxes)
    xAxes = [dataHybrid[6:9,1], dataCF[6:9,1], dataCF[6:9,1]]
    xAxes = np.array(xAxes)
    makePlot(xAxes, yAxes, lineLabels, "Artists", "Score", "Hybrid recommender for different Artist numbers with " + str(int(dataHybrid[6,0])) + " neighbours")
    
    lineLabels = ["Hybrid", "CF", "CB", "RB"]
    
    xAxes = [dataHybrid[:3,3], dataCF[:3,3], dataCB[:3,3], data[:3,3]]
    xAxes = np.array(xAxes)
    yAxes = [dataHybrid[:3,2], dataCF[:3,2], dataCB[:3,2], data[:3,2]]
    yAxes = np.array(yAxes)
    
    makePlot(xAxes, yAxes, lineLabels, "MAR", "MAP", "Relation between MAP and MAR for different recommenders with 3 neighbours (10-30 recommended Artists)")
    
    xAxes = [dataHybrid[3:6,3], dataCF[3:6,3], dataCB[3:6,3], data[3:6,3]]
    xAxes = np.array(xAxes)
    yAxes = [dataHybrid[3:6,2], dataCF[3:6,2], dataCB[3:6,2], data[3:6,2]]
    yAxes = np.array(yAxes)
    
    makePlot(xAxes, yAxes, lineLabels, "MAR", "MAP", "Relation between MAP and MAR for different recommenders with 5 neighbours (10-30 recommended Artists)")
    
    xAxes = [dataHybrid[6:9,3], dataCF[6:9,3], dataCB[6:9,3], data[6:9,3]]
    xAxes = np.array(xAxes)
    yAxes = [dataHybrid[6:9,2], dataCF[6:9,2], dataCB[6:9,2], data[6:9,2]]
    yAxes = np.array(yAxes)
    
    makePlot(xAxes, yAxes, lineLabels, "MAR", "MAP", "Relation between MAP and MAR for different recommenders with 10 neighbours (10-30 recommended Artists)")

    