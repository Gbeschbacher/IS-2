__author__ = "Eschbacher - Kratzer - Scharfetter"

import csv
import sys
import numpy as np

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
    sum_pc_user = np.sum(UAM, axis=1)
    sum_pc_artist = np.sum(UAM, axis=0)

    # Normalize the UAM (simply by computing the fraction of listening events per artist for each user)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]
    # np.tile: take sum_pc_user no_artists times (results in an array of length no_artists*no_users)
    # np.reshape: reshape the array to a matrix
    # np.transpose: transpose the reshaped matrix
    artist_sum_copy = np.tile(sum_pc_user, no_artists).reshape(no_artists, no_users).transpose()
    # Perform sum-to-1 normalization

    print artist_sum_copy
    print "\n\n\n"

    UAM = UAM / artist_sum_copy
    print UAM

    # Inform user
    print "UAM created. Users: " + str(UAM.shape[0]) + ", Artists: " + str(UAM.shape[1])

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



