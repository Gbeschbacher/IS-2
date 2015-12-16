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
if not os.path.exists('Humor-Sans.ttf'):
    fhandle = urllib2.urlopen('http://antiyawn.com/uploads/Humor-Sans-1.0.ttf')
    open('Humor-Sans.ttf', 'wb').write(fhandle.read())


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
    title = ax.set_title("\n".join(wrap(title)))

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.legend(loc='best')
    
    ax.set_xlim(float(xAxes.min() -0.5), float(xAxes.max()) + 0.5)
    ax.set_ylim(float(yAxes.min() -0.5), float(yAxes.max()) + 0.5)
    
    plt.savefig(tempTitle + ".png")
    plt.clf()

if __name__ == "__main__":
    np.random.seed(0)
    
    fileName = str(sys.argv[1])
    data = np.genfromtxt(fileName, delimiter='\t', skip_header=1)
    
    print data
    
    lineLabels = ["CF"]
    
    yAxes = data[:,2]
    xAxes = data[:,3]
    
    makePlot(xAxes, yAxes, lineLabels, "MAR", "MAP", "CF")
    
    