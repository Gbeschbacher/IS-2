__author__ = "Eschbacher - Kratzer - Scharfetter"

import csv
import sys
import numpy as np
from geopy.distance import great_circle

# name, age, country, long, lat, gender, usertype
C1KU_users = "../data/overall/C1ku_users_extended.csv"

# output
UUM_FILE = "UUM.csv"

def readUserData(filename):
    content = {}

    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = reader.next()

        for row in reader:
            user = row[0]
            content[user] = row[1:7]

    return content

# Main program
if __name__ == "__main__":

    data = readUserData(C1KU_users)
    userCount = len(data)


    # DISTANCE
    uumDistance = np.zeros(shape=(userCount, userCount), dtype=np.float32)
    for i, userOutside in enumerate(data):
        u = data[userOutside]
        positionUserOutside = np.array([0.0, 0.0])

        if u[1]:
            try:
                positionUserOutside = [float(u[2]), float(u[3])]
            except ValueError:
                print "awwwww"

        for j, userInside in enumerate(data):
            if j >= i:
                break

            u = data[userInside]
            positionUserInside = np.array([0.0, 0.0])

            if u[1]:
                try:
                    positionUserInside = [float(u[2]), float(u[3])]
                except ValueError:
                    print "awwwww"

            dist = great_circle(positionUserOutside, positionUserInside).meters
            uumDistance[i, j] = dist
            uumDistance[j, i] = dist
    uumDistance = uumDistance / uumDistance.max()
    # DISTANCE

    # AGE
    uumAge = np.zeros(shape=(userCount, userCount), dtype=np.float32)
    for i, userOutside in enumerate(data):

        u = data[userOutside]
        ageUserOutside = 0

        if int(u[0]):
            ageUserOutside = int(u[0])

        for j, userInside in enumerate(data):
            if j >= i:
                break

            u = data[userInside]
            ageUserInside = 0

            if int(u[0]):
                ageUserInside = int(u[0])

            dist = abs(ageUserInside - ageUserOutside)
            uumAge[i, j] = dist
            uumAge[j, i] = dist
    uumAge = uumAge / uumAge.max()
    #AGE

    # GENDER AND WEIGHTING
    usedGender = ["m", "f", "n"]
    uum = np.zeros(shape=(userCount, userCount), dtype=np.float32)
    for i, userOutside in enumerate(data):

        u = data[userOutside]
        genderUserOutside = "x"

        if u[4]:
            genderUserOutside = u[4]

        for j, userInside in enumerate(data):
            if j >= i:
                break

            u = data[userInside]
            genderUserInside = "x"

            if u[4]:
                genderUserInside = u[4]

            if genderUserInside in usedGender and genderUserOutside in usedGender:
                dist = 1
            
                if genderUserInside == genderUserOutside:
                    dist = 0
                
                dissimilarity = uumAge[i, j] * 0.33 + uumDistance[i, j] * 0.33 + dist * 0.33
            dissimilarity = uumAge[i, j] * 0.33 + uumDistance[i, j] * 0.33
            uum[i, j] = 1 - dissimilarity
            uum[j, i] = 1 - dissimilarity
    # GENDER AND WEIGHTING

    np.savetxt("../data/overall/UUM.csv", uum, fmt="%0.6f", delimiter="\t", newline="\n")
