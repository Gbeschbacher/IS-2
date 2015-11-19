__author__ = "Eschbacher - Kratzer - Scharfetter"

import lfm
import helpers

import json
import sys

import time

uniqueUsers = []
countryDict = {}

MINIMUM_PLAYCOUNT = 1000
MAXIMUM_USER_PRO_COUNTRY = 50


def isUserRelevant(user):

    if( "country" in user and "playcount" in user):

        countryName = str(user["country"])

        numberUsersInCountry = 0
        if countryName in countryDict:
            numberUsersInCountry = countryDict[countryName]

        return countryName != "" and int(user["playcount"]) >= MINIMUM_PLAYCOUNT and numberUsersInCountry < MAXIMUM_USER_PRO_COUNTRY and user not in uniqueUsers
    else:
        return False


if __name__ == "__main__":
    pathToBase = str(sys.argv[1])
    pathToOverall = str(sys.argv[2])
    minimumUsers = int(sys.argv[3])

    start = time.time()

    users = helpers.readFile(pathToBase)

    for i, user in enumerate(users):

        print "\n"
        print "Process user ", i
        print "Unique users", len(uniqueUsers)
        print "Users to process", len(users)

        if len(uniqueUsers) >= minimumUsers:
            break

        content = lfm.call("user.getInfo", params = {"user": user})
        userInfoTree = json.loads(content)

        if not "user" in userInfoTree:
            continue

        userInfo = userInfoTree["user"]

        if not isUserRelevant(userInfo):
            print "User is not relevant"
            continue

        print "User is relevant"

        uniqueUsers.append(userInfo)
        country = str(userInfo["country"])
        if not country in countryDict:
            countryDict[country] = 0
        countryDict[country] += 1

        friends = lfm.call("user.getFriends", params={"user": user})
        friendsTree = json.loads(friends)

        if not "friends" in friendsTree:
            print "User has no friends"
            continue

        if len(friendsTree["friends"]["user"]) > 2:
            firstFriend = str(friendsTree["friends"]["user"][0]["name"])
            secondFriend = str(friendsTree["friends"]["user"][1]["name"])

            users.append(firstFriend)
            users.append(secondFriend)

    end = time.time()

    print ("Program took {0} seconds").format(end - start)

    filename = "unique_users_{0}.csv".format(minimumUsers)
    with open(pathToOverall + filename, "w") as outfile:
        outfile.write("name\tcountry\tage\tgender\tplaycount\ttype\n")

        for uniqueUser in uniqueUsers:
            if "gender" in uniqueUser:
                gender = uniqueUser["gender"]
            else:
                gender = "n/a"

            outfile.write(uniqueUser["name"] + "\t" + uniqueUser["country"] + "\t" + uniqueUser["age"] + "\t" + gender + "\t" + uniqueUser["playcount"] + "\t" + uniqueUser["type"] + "\n")

    filename = "country_dictionary_{0}.csv".format(minimumUsers)
    with open(pathToOverall + filename, "w") as outfile:
        outfile.write("country\tnumber of users\n")
        for country, count in countryDict.iteritems():
            outfile.write(country + "\t" + str(count) + "\n")

