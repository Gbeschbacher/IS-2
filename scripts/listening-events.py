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

    for i, user in enumerate(users):
        print "Processing LEs for user ", i

        userContent = []

        for page in range(1,6):
            content = lfm.call("user.getRecentTracks", params = {"user": user, "limit": 200, "page": page})

            userContent.append(content)
            listeningEventsTree = json.loads(content)

            if not checkEventTree(listeningEventsTree):
                continue

            for event in listeningEventsTree["recenttracks"]["track"]:
                artistID = event["artist"]["mbid"]

                if (artistID in uniqueArtists or artistID == ""):
                    continue

                if not "artist" in event:
                    continue
                if not "name" in event:
                    continue
                if not "date" in event:
                    continue

                artist = event["artist"]["#text"]
                track = event["name"]
                time = event["date"]["uts"]

                LEs.append([user, artist.encode("utf8"), track.encode("utf8"), str(time)])
                uniqueArtists.append(artistID)


    filename = "listening_events_{0}.csv".format(len(LEs))
    with open(pathToOverall + filename, "w") as outfile:
        outfile.write("user\tartist\ttrack\ttime\n")
        for le in LEs:
            outfile.write(le[0] + "\t" + le[1] + "\t" + le[2] + "\t" + le[3] + "\n")

    filename = "unique_artists_{0}.csv".format(len(uniqueArtists))
    with open(pathToOverall + filename, "w") as outfile:
        for uniqueArtist in uniqueArtists:
            outfile.write(uniqueArtist + "\n")
