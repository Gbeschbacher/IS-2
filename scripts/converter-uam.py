__author__ = "Eschbacher - Kratzer - Scharfetter"

import csv
import sys
import numpy as np

MIN_ARTISTS_COUNT_PER_USER = 10
MIN_USER_COUNT_PER_ARTIST = 5

def filter_out_users(users, artists, listening_events, min_artists_count):
    anythingDeleted = False
    usersToDelete = []
    for user in users.keys():
        artistCount = 0
        for artist in artists.keys():
            if((user, artist) in listening_events):
                if(listening_events[(user, artist)] > 0):
                    artistCount += 1
        if(artistCount < min_artists_count):
            usersToDelete.append(user)
            for artist in artists.keys():
                if((user, artist) in listening_events):
                    del(listening_events[(user, artist)])

    for user in usersToDelete:
        del(users[user])
        anythingDeleted = True

    # remove all artists that no user is connected with at all
    artistsToDelete = []
    for artist in artists.keys():
        removeArtist = True
        for user in users.keys():
            if (user, artist) in listening_events:
                removeArtist = False
                break
        if(removeArtist):
            artistsToDelete.append(artist)

    for artist in artistsToDelete:
        del(artists[artist])
        anythingDeleted = True

    return anythingDeleted

def filter_out_artists(users, artists, listening_events, min_user_count):
    anythingDeleted = False
    artistsToDelete = []
    for artist in artists.keys():
        userCount = 0
        for user in users.keys():
            if((user, artist) in listening_events):
                if(listening_events[(user, artist)] > 0):
                    userCount += 1
        if(userCount < min_user_count):
            artistsToDelete.append(artist)
            for user in users.keys():
                if((user, artist) in listening_events):
                    del(listening_events[(user, artist)])

    for artist in artistsToDelete:
        del(artists[artist])
        anythingDeleted = True

    # remove all users that no artists is connected with at all
    usersToDelete = []
    for user in users.keys():
        removeUser = True
        for artist in artists.keys():
            if (user, artist) in listening_events:
                removeUser = False
                break
        if(removeUser):
            usersToDelete.append(user)

    for user in usersToDelete:
        del(users[user])
        anythingDeleted = True

    return  anythingDeleted

if __name__ == "__main__":

    pathToListeningEvents = str(sys.argv[1])
    pathToOverall = str(sys.argv[2])

    # dictionary, (mis)used as ordered list of artists without duplicates
    artists = {}

    # dictionary, (mis)used as ordered list of users without duplicates
    users = {}

    # dictionary to store assignments between user and artist
    listening_events = {}

    # Read listening events from provided file
    with open(pathToListeningEvents, "r") as f:

        # create reader
        reader = csv.reader(f, delimiter="\t")
        # skip header
        headers = reader.next()
        for row in reader:
            user = row[0]
            artist = row[1]
            track = row[2]
            time = row[3]

            # create ordered set (list) of unique elements (for artists / tracks)
            artists[artist] = None
            users[user] = None

            # initialize listening event counter, access by tuple (user, artist) in dictionary
            listening_events[(user, artist)] = 0


    # Read listening events from provided file (to fill user-artist matrix)
    with open(pathToListeningEvents, "r") as f:
        # create reader
        reader = csv.reader(f, delimiter="\t")
        # skip header
        headers = reader.next()
        for row in reader:
            user = row[0]
            artist = row[1]
            track = row[2]
            time = row[3]
            # increase listening counter for (user, artist) pair/tuple
            listening_events[(user, artist)] += 1

    print "artist count: " + len(artists).__str__()
    print "user count: " + len(users).__str__()
    print "listenings count: " + len(listening_events).__str__()

    count = 0
    while(True):
        count += 1
        # filter out all artists wth less than "x" users
        artistsDeleted = filter_out_artists(users, artists, listening_events, MIN_USER_COUNT_PER_ARTIST)

        #print "artist count: " + len(artists).__str__()
        #print "user count: " + len(users).__str__()
        #print "listenings count: " + len(listening_events).__str__()

        # filter out all users with less than "x" artists
        usersDeleted = filter_out_users(users, artists, listening_events, MIN_ARTISTS_COUNT_PER_USER)

        if(not artistsDeleted and not usersDeleted):
            break

    print ("filtering count: " + count.__str__())
    print "artist count: " + len(artists).__str__()
    print "user count: " + len(users).__str__()
    print "listenings count: " + len(listening_events).__str__()


    # Assign a unique index to all artists and users in dictionary (we need these to create the UAM)
    counter = 0
    for artist in artists.keys():
        artists[artist] = counter
        counter += 1

    counter = 0
    for user in users.keys():
        users[user] = counter
        counter += 1

    # Now we use numpy to create the UAM
    # first, create an empty matrix
    UAM = np.zeros(shape=(len(users.keys()), len(artists.keys())), dtype="float32")

    # iterate through all (user, artist) tuples in listening_events
    for u in users.keys():
        for a in artists.keys():
            try:
                # get correct index for user u and artist a
                idx_u = users[u]
                idx_a = artists.get(a)

                # insert number of listening events of user u to artist a in UAM
                UAM[idx_u, idx_a] = listening_events[(u,a)]

            # if user u did not listen to artist a, we continue
            except KeyError:
                continue

    # Get sum of play events per user and per artist
    # sum_pc_user = np.sum(UAM, axis=1)
    # sum_pc_artist = np.sum(UAM, axis=0)

    # Normalize the UAM (simply by computing the fraction of listening events per artist for each user)
    # no_users = UAM.shape[0]
    # no_artists = UAM.shape[1]
    # # np.tile: take sum_pc_user no_artists times (results in an array of length no_artists*no_users)
    # # np.reshape: reshape the array to a matrix
    # # np.transpose: transpose the reshaped matrix
    # artist_sum_copy = np.tile(sum_pc_user, no_artists).reshape(no_artists, no_users).transpose()
    # Perform sum-to-1 normalization

    for u in range (0, UAM.shape[0]):
        lineVec = UAM[u,:]
        lineVecNorm = 0
        for v in lineVec:
            lineVecNorm += v*v
        lineVecNorm = np.sqrt(lineVecNorm)
        UAM[u,:] = UAM[u,:] / lineVecNorm

    # Write everything to text file (artist names, user names, UAM)
    # Write artists to text file
    filename = "UAM_artists.csv"
    with open(pathToOverall + filename, "w") as outfile:
        outfile.write("artist\n")
        for key in artists.keys():
            outfile.write(key + "\n")
    outfile.close()

    filename = "UAM_users.csv"
    with open(pathToOverall + filename, "w") as outfile:
        outfile.write("user\n")
        for key in users.keys():
            outfile.write(key + "\n")
    outfile.close()

    # Write UAM
    filename = pathToOverall + "UAM.csv"
    np.savetxt(filename, UAM, fmt="%0.6f", delimiter="\t", newline="\n")



