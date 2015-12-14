__author__ = "Eschbacher - Kratzer - Scharfetter"

import csv
import os
import urllib

#Reading a file and returning its content
def readFile(filename, header = False, rowIndex = 0):
    content = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        if header:
            headers = reader.next()

        for row in reader:
            content.append(row[rowIndex])

    return content

#Reading a path on the os if it doesn't exist
def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

# A simple function to remove HTML tags from a string.
def removeHTMLMarkup(s):
    tag = False
    quote = False
    out = ""

    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out
