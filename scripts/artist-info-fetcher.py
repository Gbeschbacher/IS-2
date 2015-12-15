__author__ = "Eschbacher - Kratzer - Scharfetter"

# Load required modules
import os
import sys
import helpers
import Queue
import threading
import json
import urllib
import urllib2

WIKIPEDIA_URL_SS = "http://en.wikipedia.org/wiki/Special:Search/"
THEAUDIODB_URL = "http://www.theaudiodb.com/artist/"
DISCOGS_URL = "http://www.discogs.com/artist/"

fetcher = [WIKIPEDIA_URL_SS, THEAUDIODB_URL, DISCOGS_URL]

def getUrl(q, url):
    print url
    try:
        q.put(urllib2.urlopen(url).read())
    except:
        print "error"

def getArtistInfo(artist):
    contentMerged = []
    q = Queue.Queue()
    
    threads = []

    for j, baseUrl in enumerate(fetcher):
        url = baseUrl + urllib2.quote(artist)

        t = threading.Thread(target=getUrl, args = (q,url))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
        
    for k in range(0, q.qsize()):
        contentMerged.append(q.get(False))

    return contentMerged

# Main program
if __name__ == '__main__':

    pathToArtistInfo = str(sys.argv[1])
    pathToUniqueArtists = str(sys.argv[2])

    helpers.createPath(pathToArtistInfo)

    artists = helpers.readFile(pathToUniqueArtists, True, 1)

    for i in range(0, len(artists)):
        print "Fetching artist " + str(i)
        htmlFn = pathToArtistInfo + str(i) + ".html"
        if os.path.exists(htmlFn):
            print "File already fetched: " + htmlFn
            continue

        htmlContent = getArtistInfo(artists[i])
        print "Storing content to " + htmlFn
        with open(htmlFn, 'w') as f:
            f.write("".join(htmlContent))
