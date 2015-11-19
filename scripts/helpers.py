__author__ = "Eschbacher - Kratzer - Scharfetter"

import csv
import os

#Reading a file and returning its content
def readFile(filename, header = False):
    content = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        if header:
            headers = reader.next()

        for row in reader:
            content.append(row[0])

    return content

#Reading a path on the os if it doesn't exist
def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
