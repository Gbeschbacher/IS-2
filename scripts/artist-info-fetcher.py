__author__ = "Eschbacher - Kratzer - Scharfetter"

# Load required modules
import os
import sys
import helpers

WIKIPEDIA_URL_SS = "http://en.wikipedia.org/wiki/Special:Search/"
THEAUDIODB_URL = "http://www.theaudiodb.com/artist/"
DISCOGS_URL = "http://www.discogs.com/artist/"

# Main program
if __name__ == '__main__':

    pathToArtistInfo = str(sys.argv[1])
    pathToUniqueArtists = str(sys.argv[2])

    helpers.createPath(pathToArtistInfo)

    artists = helpers.readFile(pathToUniqueArtists)

    for i in range(0, len(artists)):
        htmlFn = pathToArtistInfo + "/" + str(i) + ".html"
        if os.path.exists(htmlFn):
            print "File already fetched: " + htmlFn
            continue

        htmlContent = helpers.fetchWebPage(WIKIPEDIA_URL_SS, artists[i])
        htmlContent += helpers.fetchWebPage(THEAUDIODB_URL, artists[i])
        htmlContent += helpers.fetchWebPage(DISCOGS_URL, artists[i])

        print "Storing content to " + htmlFn
        with open(htmlFn, 'w') as f:
            f.write(htmlContent)
