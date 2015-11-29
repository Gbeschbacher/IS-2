__author__ = "Eschbacher - Kratzer - Scharfetter"

import lfm
import helpers

import json
import sys

uniqueArtists = []
LEs = []

def checkEventTree(eventTree):

    if not "recenttracks" in eventTree:
        print "Recent tracks not found"
        return False

    if not "track" in listeningEventsTree["recenttracks"]:
        print "Tracks not found"
        return False

    return True

if __name__ == "__main__":
    pathToUniqueUsers = str(sys.argv[1])
    pathToOverall = str(sys.argv[2])

    users = helpers.readFile(pathToUniqueUsers, header = True)

    LEs = []

    for user in range(0, len(users)):
        print "Processing LEs for user ", user

        content = lfm.getLEs(users[user], 5, 200)

        try:
            # for all retrieved JSON pages of current user
            for page in range(0, len(content)):
                listening_events = json.loads(content[page])

                # get number of listening events in current JSON
                numberOfItems = 0
                # catching possible bug when 0 tracks are returned by listening_events
                if "recenttracks" in listening_events:
                    if "track" in listening_events["recenttracks"]:
                        numberOfItems = len(listening_events["recenttracks"]["track"])
                print "number items: " + numberOfItems.__str__()

                # read artist and track names for each
                for item in range(0, numberOfItems):
                    artist = listening_events["recenttracks"]["track"][item]["artist"]["#text"]
                    track = listening_events["recenttracks"]["track"][item]["name"]
                    time = listening_events["recenttracks"]["track"][item]["date"]["uts"]

                    LEs.append([users[user], artist.encode('utf8'), track.encode('utf8'), str(time)])

        except KeyError:                    # JSON tag not found
            print "JSON tag not found!"
            continue

    filename = "listening_events_{0}.csv".format(len(LEs))
    with open(pathToOverall + filename, "w") as outfile:
        outfile.write("user\tartist\ttrack\ttime\n")
        for le in LEs:
            outfile.write(le[0] + "\t" + le[1] + "\t" + le[2] + "\t" + le[3] + "\n")

    filename = "unique_artists_{0}.csv".format(len(uniqueArtists))
    with open(pathToOverall + filename, "w") as outfile:
        for uniqueArtist in uniqueArtists:
            outfile.write(uniqueArtist + "\n")
