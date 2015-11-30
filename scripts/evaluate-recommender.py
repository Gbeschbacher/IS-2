# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# contet-based recommender, random recommendation, and simple hybrid methods.
# It also implements a score-based fusion technique for hybrid recommendation.
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
import operator
from operator import itemgetter                 # for sorting dictionaries w.r.t. values
import sys

# Parameters
UAM_FILE = "../data/overall/UAM.csv"                # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/overall/UAM_artists.csv"    # artist names for UAM
USERS_FILE = "../data/overall/UAM_users.csv"        # user names for UAM
AAM_FILE = "../data/overall/aam.csv"                # artist-artist similarity matrix (AAM)
METHOD = "HR_SCB"                       # recommendation method
                                    # ["RB", "CF", "CB", "HR_SEB", "HR_SCB"]
K = 3           # for CB: number of nearest neighbors to consider for each artist in seed user's training set
MAX_ARTISTS = 30          # for hybrid: number of artists to recommend at most

NF = 10              # number of folds to perform in cross-validation


# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter='\t')  # create reader
        headers = reader.next()  # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    f.close()
    return data


def recommend_CF(UAM, seed_uidx, seed_aidx_train, seed_aidx_test, K, max_artists):
    # UAM               user-artist-matrix
    # seed_uidx         user index of seed user
    # seed_aidx_train   indices of training artists for seed user

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Set to 0 the listening events of seed user user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, seed_aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx,:] = UAM[seed_uidx,:] / np.sum(UAM[seed_uidx,:])


    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    #sim_users = np.inner(pc_vec, UAM)  # similarities between u and other users

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_indices = sort_idx[-1-K:-1]

#    print "The closest user to user " + str(seed_uidx) + " is " + str(neighbor_idx) + "."
#    print "The closest user to user " + users[seed_uidx] + " is user " + users[neighbor_idx] + "."

    # Get all artist indices the seed user and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = seed_aidx_train

    recommended_artists_idx_all_dic = {}
    x = 1
    y = 1
    for neighbor_idx in neighbor_indices:
        # Compute the set difference between seed user's neighbor and seed user,
        # i.e., artists listened to by the neighbor, but not by seed user.
        # These artists are recommended to seed user.

        artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to
        # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
        recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u)

        for idx in recommended_artists_idx:
            if(not recommended_artists_idx_all_dic.has_key(idx)):
                recommended_artists_idx_all_dic[idx] = y
            else:
                recommended_artists_idx_all_dic[idx] += y

        y = (x * x) / 2 + 1
        x += 1

    sorted_recommended_artists_key_value = sorted(recommended_artists_idx_all_dic.items(), key=operator.itemgetter(1), reverse = True)
    max_value = sorted_recommended_artists_key_value[0][1]
    recommended_artists_idx_all = {}
    count = 1
    for (key, value) in sorted_recommended_artists_key_value:
        recommended_artists_idx_all[key] = float(float(value) / float(max_value))
        if count >= max_artists:
            break
        count += 1

    # Return list of recommended artist indices
    return recommended_artists_idx_all


# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, K, max_artists):
    # AAM               artist-artist-matrix of pairwise similarities
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (artists) to consider for each seed artist


    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train,:], axis=1)

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:,-1-K:-1]

    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}           # dictionry to hold recommended artists and corresponding scores

    # Distill corresponding similarity scores and store in sims_neighbors_idx
    sims_neighbors_idx = np.zeros(shape=(len(seed_aidx_train), K), dtype=np.float32)
    for i in range(0, neighbor_idx.shape[0]): # 0 = y-axis. 1 = x-axis
        sims_neighbors_idx[i] = AAM[seed_aidx_train[i], neighbor_idx[i]]

    # Aggregate the artists in neighbor_idx.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(neighbor_idx.flatten())     # First, we obtain a unique set of artists neighboring the seed user's artists.
    # Now, we find the positions of each unique neighbor in neighbor_idx.
    # max_length = 0
    for nidx in uniq_neighbor_idx:
        mask = np.where(neighbor_idx == nidx)
        sims_list_for_curr_nidx = sims_neighbors_idx[mask]

        sum_sim = np.sum(sims_list_for_curr_nidx)
        # if(len(sims_list_for_curr_nidx) > max_length):
        #     max_length = len(sims_list_for_curr_nidx)
        dict_recommended_artists_idx[nidx] = sum_sim

        # Apply this mask to corresponding similarities and compute average similarity
        # avg_sim = np.mean(sims_list_for_curr_nidx)
        # Store artist index and corresponding aggregated similarity in dictionary of arists to recommend
        # dict_recommended_artists_idx[nidx] = avg_sim
    #########################################

    # Remove all artists that are in the training set of seed user
    for aidx in seed_aidx_train:
        dict_recommended_artists_idx.pop(aidx, None)            # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    # normalize similarity values in dict_recommended_artists_idx
    # for key, value in dict_recommended_artists_idx.items():
    #     dict_recommended_artists_idx[key] = value / max_length

    sorted_recommended_artists_key_value = sorted(dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse = True)
    max_value = sorted_recommended_artists_key_value[0][1]
    recommended_artists_idx_all = {}
    count = 1
    for (key, value) in sorted_recommended_artists_key_value:
        recommended_artists_idx_all[key] = float(float(value) / float(max_value))
        if count >= max_artists:
            break
        count += 1

    # Return dictionary of recommended artist indices (and scores)
    return recommended_artists_idx_all


# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx

def run(artists, users, UAM, AAM) :

   # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall
    
    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]
    for u in range(0, no_users):
    
        #print ("User: %f0" % (u))

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        for train, test in kf:  # for all folds

            test_aidx = u_aidx[test]
            train_aidx = u_aidx[train]

            # Show progress
            #print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
            #    len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable


            # Run recommendation method specified in METHOD
            # NB: u_aidx[train_aidx] gives the indices of training artists
            rec_aidx = {}   # use a dictionary to store (similarity) scores along recommended artist indices (in contrast to Evaluate_Recommender.py)


            if METHOD == "RB":          # random baseline
                dict_rec_aidx = recommend_RB(np.setdiff1d(range(0, no_artists), train_aidx), MAX_ARTISTS) # len(test_aidx))
            elif METHOD == "CF":        # collaborative filtering
                dict_rec_aidx = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
            elif METHOD == "CB":        # content-based recommender
                dict_rec_aidx = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
            elif METHOD == "HR_SCB":     # hybrid of CF and CB, using score-based fusion (SCB)
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                # Fuse scores given by CF and by CB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_CF.keys():
                    # scores[1, aidx] = dict_rec_aidx_CF[aidx] * 0.75 # 3.45 / 11.30
                    scores[1, aidx] = dict_rec_aidx_CF[aidx] * 0.5 # 3.72 / 12.14
                    # scores[1, aidx] = dict_rec_aidx_CF[aidx] * dict_rec_aidx_CF[aidx] # 3.48 / 11.41
                # Apply aggregation function
                scores_fused = np.max(scores, axis=0)
                # Sort and select top artists to recommend
                sorted_idx = np.argsort(scores_fused)
                # artists_length = min(len(dict_rec_aidx_CB.items()) + len(dict_rec_aidx_CF.items()), MAX_ARTISTS)
                artists_length = min(len(dict_rec_aidx_CB.items()), len(dict_rec_aidx_CF.items()))
                sorted_idx_top = sorted_idx[len(sorted_idx)-artists_length:len(sorted_idx)]

                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]

                # Alternative code to select top K_HR artists
                # # To this end, sort recommended artists in descending order and return top N
                # dict_rec_aidx_tmp = {}
                # vs = sorted(dict_rec_aidx.items(), key=itemgetter(1), reverse=True)
                # vs = vs[:K_HR]
                # for i in range(0, len(vs)):
                #     dict_rec_aidx_tmp[vs[i][0]] = vs[i][1]
                # dict_rec_aidx = dict_rec_aidx_tmp


            # Distill recommended artist indices from dictionary returned by the recommendation functions
            rec_aidx = dict_rec_aidx.keys()

            #print "Recommended items: ", len(rec_aidx)
            #print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(test_aidx, rec_aidx)          # correctly predicted artists
            # True Positives is amount of overlap in recommended artists and test artists
            TP = len(correct_aidx)
            # False Positives is recommended artists minus correctly predicted ones
            FP = len(np.setdiff1d(rec_aidx, correct_aidx))
            # Precision is percentage of correctly predicted among predicted
            if len(rec_aidx) == 0:      # if we cannot predict a single artist -> precision is defined as 1 (we don't make wrong predictions)
                prec = 100.0
            else:               # compute precision
                prec = 100.0 * TP / len(rec_aidx)
            # Recall is percentage of correctly predicted among all listened to
            rec = 100.0 * TP / len(test_aidx)


            # add precision and recall for current user and fold to aggregate variables
            avg_prec += prec / (NF * no_users)
            avg_rec += rec / (NF * no_users)

            # Output precision and recall of current fold
            #print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    # Output mean average precision and recall
    fScore = 2 * (avg_prec * avg_rec)/(avg_prec + avg_rec)
    print ("\n%.0f, %.0f %.2f, %.2f, %.2f" % (K, MAX_ARTISTS, avg_prec, avg_rec, fScore))



# Main program
if __name__ == '__main__':

    ARTISTS_FILE = str(sys.argv[3])
    USERS_FILE = str(sys.argv[4])
    UAM_FILE = str(sys.argv[1])
    AAM_FILE = str(sys.argv[2])
    METHOD = str(sys.argv[5])
    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
    # Load AAM
    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    artistSteps = [10,20,30]
    neighbourSteps = [3,5,10]
    print "Neighbours, Recommended Artists, MAP, MAR, FSCORE"
    for k in neighbourSteps :
        K = k
        for a in artistSteps :
            MAX_ARTISTS = a
            run(artists, users, UAM, AAM)
    