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
from time import time
from multiprocessing import Process

import os

# Parameters
UAM_FILE = "../data/overall/C1ku_UAM.csv"                # user-artist-matrix (UAM)
UUM_FILE = "../data/overall/UUM.csv"                # user-artist-matrix (UAM)
ARTISTS_FILE = "../data/overall/C1ku_artists_extended.csv"    # artist names for UAM
USERS_FILE = "../data/overall/C1ku_users_extended.csv"        # user names for UAM
AAM_FILE = "../data/overall/aam.csv"                # artist-artist similarity matrix (AAM)



NF = 5              # number of folds to perform in cross-validation


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


def recommend_PB(UAM, seed_aidx_train, max_artists):
    # UAM               user-artist-matrix
    # seed_aidx_train   indices of training artists for seed user (to exclude corresponding recommendations)
    # K                 number of artists to recommend

    # Remove training set artists from UAM (we do not want to include these in the recommendations)
    UAM[:,seed_aidx_train] = 0.0

    # Ensure that number of available artists is not smaller than number of requested artists (excluding training set artists)
    no_artists = UAM.shape[1]
    if max_artists > no_artists - len(seed_aidx_train):
        print str(max_artists) + " artists requested, but dataset contains only " + str(no_artists) + " artists! Reducing number of requested artists to " + str(no_artists) + "."
        max_artists = no_artists - len(seed_aidx_train)

    # get max_artists most popular artists, according to UAM
    UAM_sum = np.sum(UAM, axis=0)                                    # sum all (normalized) listening events per artist
    popsorted_aidx = np.argsort(UAM_sum)[-max_artists:]                    # indices of popularity-sorted artists (max_artists most popular artists)
    recommended_artists_idx = popsorted_aidx                         # artist indices
    recommended_artists_scores = UAM_sum[popsorted_aidx]             # corresponding popularity scores

    # Normalize popularity scores to range [0,1], to enable fusion with other approaches
    recommended_artists_scores = recommended_artists_scores / np.max(recommended_artists_scores)

    # Insert indices and scores into dictionary
    dict_recommended_artists_idx = {}
    for i in range(0, len(recommended_artists_idx)):
        dict_recommended_artists_idx[recommended_artists_idx[i]] = recommended_artists_scores[i]
#        print artists[recommended_artists_idx[i]] + ": " + str(recommended_artists_scores[i])

    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx


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

    recommended_artists_indices = {}
    while(True):
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

        if(len(recommended_artists_idx_all_dic) >= max_artists):
            recommended_artists_indices = recommended_artists_idx_all_dic
            break
        else:
            K += 1

    sorted_recommended_artists_key_value = sorted(recommended_artists_indices.items(), key=operator.itemgetter(1), reverse = True)
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


def recommend_UBCF(UAM, UUM, seed_uidx, seed_aidx_train, seed_aidx_test, K, max_artists):
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
        sim = 1.0 - scidist.cosine(pc_vec, UAM[u,:])
        sim_users[u] = sim * UUM[seed_uidx, u]
        # if UUM[seed_uidx, u] < 0.7:
        #     sim_users[u] = 0

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    #sim_users = np.inner(pc_vec, UAM)  # similarities between u and other users

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    recommended_artists_indices = {}
    while(True):
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

        if(len(recommended_artists_idx_all_dic) >= max_artists):
            recommended_artists_indices = recommended_artists_idx_all_dic
            break
        else:
            K += 1

    sorted_recommended_artists_key_value = sorted(recommended_artists_indices.items(), key=operator.itemgetter(1), reverse = True)
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


def recommend_UB(UAM, UUM, seed_uidx, seed_aidx_train, K, max_artists):
    unordered_current_user_similarities = UUM[seed_uidx,:]
    sorted_idx = np.argsort(unordered_current_user_similarities)

    recommended_artists_indices = {}
    while(True):
        neighbor_indices = sorted_idx[-1-K:]

        artist_idx_u = seed_aidx_train
        recommended_artists_idx_all_dic = {}
        for neighbor_idx in neighbor_indices:
            # Compute the set difference between seed user's neighbor and seed user,
            # i.e., artists listened to by the neighbor, but not by seed user.
            # These artists are recommended to seed user.

            if(neighbor_idx != seed_uidx):
                artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to
                # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
                recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u)

                for idx in recommended_artists_idx:
                    if(not recommended_artists_idx_all_dic.has_key(idx)):
                        recommended_artists_idx_all_dic[idx] = unordered_current_user_similarities[neighbor_idx]
                    else:
                        recommended_artists_idx_all_dic[idx] += unordered_current_user_similarities[neighbor_idx]

        if(len(recommended_artists_idx_all_dic) >= max_artists):
            recommended_artists_indices = recommended_artists_idx_all_dic
            break
        else:
            K += 1

    sorted_recommended_artists_key_value = sorted(recommended_artists_indices.items(), key=operator.itemgetter(1), reverse = True)
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

    aidx_train = [x for x in seed_aidx_train if not x == 10121]

    sort_idx = np.argsort(AAM[aidx_train,:], axis=1)

    recommended_artists_indices = {}
    while(True):
        # Select the K closest artists to all artists the seed user listened to
        neighbor_idx = sort_idx[:,-1-K:-1]

        ##### ADDED FOR SCORE-BASED FUSION  #####
        dict_recommended_artists_idx = {}           # dictionry to hold recommended artists and corresponding scores

        # Distill corresponding similarity scores and store in sims_neighbors_idx
        sims_neighbors_idx = np.zeros(shape=(len(aidx_train), K), dtype=np.float32)
        for i in range(0, neighbor_idx.shape[0]): # 0 = y-axis. 1 = x-axis
            sims_neighbors_idx[i] = AAM[aidx_train[i], neighbor_idx[i]]

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
        for aidx in aidx_train:
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

        if(count >= max_artists):
            recommended_artists_indices = recommended_artists_idx_all
            break
        else:
            K += 1

    # Return dictionary of recommended artist indices (and scores)
    return recommended_artists_indices


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

	# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RBU(artists_idx, no_items, UAM, user):
    # artists_idx           list of artist indices in the training set
    # no_items              no of items to predict
    
    random_uidx = random.sample(np.setdiff1d(range(0, UAM.shape[0]), [user]), 10)
    
    listened = []
    for uidx in random_uidx:
      listened.extend(np.nonzero(UAM[random_uidx, :])[0])

    listened = np.setdiff1d(list(set(listened)), artists_idx)
    
    dict_random_aidx = {}
    for aidx in random.sample(listened, min(len(listened), no_items)):
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal
            

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx



def run(artists, users, UAM, UUM, AAM, no_users, no_artists, METHOD, K, MAX_ARTISTS):
    foldername = "./results/"
    filename = foldername + str(METHOD) + "_k.txt"

    # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall

    t0 = time()

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]
        if(len(u_aidx) < NF) :
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        for train, test in kf:  # for all folds

            test_aidx = u_aidx[test]
            train_aidx = u_aidx[train]

            # Show progress
            # print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
            #     len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable


            # Run recommendation method specified in METHOD
            # NB: u_aidx[train_aidx] gives the indices of training artists
            rec_aidx = {}   # use a dictionary to store (similarity) scores along recommended artist indices (in contrast to Evaluate_Recommender.py)

            if METHOD == "RB":          # random baseline
                dict_rec_aidx = recommend_RB(np.setdiff1d(range(0, no_artists), train_aidx), MAX_ARTISTS) # len(test_aidx))
            if METHOD == "RBU":          # random baseline
                dict_rec_aidx = recommend_RBU(train_aidx, MAX_ARTISTS, copy_UAM, u) # len(test_aidx))
            elif METHOD == "CF":        # collaborative filtering
                dict_rec_aidx = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
            elif METHOD == "UB":
                dict_rec_aidx = recommend_UB(copy_UAM, UUM, u, train_aidx, K, MAX_ARTISTS)
            elif METHOD == "UBCF":        # collaborative filtering
                dict_rec_aidx = recommend_UBCF(copy_UAM, UUM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
            elif METHOD == "CB":        # content-based recommender
                dict_rec_aidx = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
            elif METHOD == "PB":
                dict_rec_aidx = recommend_PB(copy_UAM, train_aidx, MAX_ARTISTS)
            elif METHOD == "HR_CBCF_SB":     # hybrid of CF and CB, using score-based fusion (SCB)
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
            elif METHOD == "HR_UBCF_SB":     # hybrid of CF and CB, using score-based fusion (SCB)
                dict_rec_aidx_UB = recommend_UB(copy_UAM, UUM, u, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                # Fuse scores given by CF and by CB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_UB.keys():
                    scores[0, aidx] = dict_rec_aidx_UB[aidx] * 0.2
                for aidx in dict_rec_aidx_CF.keys():
                    # scores[1, aidx] = dict_rec_aidx_CF[aidx] * 0.75 # 3.45 / 11.30
                    scores[1, aidx] = dict_rec_aidx_CF[aidx] # * 0.5 # 3.72 / 12.14
                    # scores[1, aidx] = dict_rec_aidx_CF[aidx] * dict_rec_aidx_CF[aidx] # 3.48 / 11.41
                # Apply aggregation function
                scores_fused = np.max(scores, axis=0)
                # Sort and select top artists to recommend
                sorted_idx = np.argsort(scores_fused)
                # artists_length = min(len(dict_rec_aidx_CB.items()) + len(dict_rec_aidx_CF.items()), MAX_ARTISTS)
                artists_length = min(len(dict_rec_aidx_UB.items()), len(dict_rec_aidx_CF.items()))
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
            elif METHOD == "HR_CBPB_SB":     # hybrid of CF and CB, using score-based fusion (SCB)
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_PB = recommend_PB(copy_UAM, train_aidx, MAX_ARTISTS)
                # Fuse scores given by CF and by CB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_PB.keys():
                    # scores[1, aidx] = dict_rec_aidx_CF[aidx] * 0.75 # 3.45 / 11.30
                    scores[1, aidx] = dict_rec_aidx_PB[aidx] * 0.4  # 3.72 / 12.14
                    # scores[1, aidx] = dict_rec_aidx_CF[aidx] * dict_rec_aidx_CF[aidx] # 3.48 / 11.41
                # Apply aggregation function
                scores_fused = np.max(scores, axis=0)
                # Sort and select top artists to recommend
                sorted_idx = np.argsort(scores_fused)
                # artists_length = min(len(dict_rec_aidx_CB.items()) + len(dict_rec_aidx_CF.items()), MAX_ARTISTS)
                artists_length = min(len(dict_rec_aidx_CB.items()), len(dict_rec_aidx_PB.items()))
                sorted_idx_top = sorted_idx[len(sorted_idx)-artists_length:len(sorted_idx)]

                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]

            elif METHOD == "HR_CBUBCF_SB":     # hybrid of CF and CB, using score-based fusion (SCB)
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_UB = recommend_UB(copy_UAM, UUM, u, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                # Fuse scores given by CF and by CB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(3, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_CF.keys():
                    scores[1, aidx] = dict_rec_aidx_CF[aidx] * 0.5  # 3.72 / 12.14
                for aidx in dict_rec_aidx_UB.keys():
                    scores[2, aidx] = dict_rec_aidx_UB[aidx] * 0.2  # 3.72 / 12.14
                # Apply aggregation function
                scores_fused = np.max(scores, axis=0)
                # Sort and select top artists to recommend
                sorted_idx = np.argsort(scores_fused)
                # artists_length = min(len(dict_rec_aidx_CB.items()) + len(dict_rec_aidx_CF.items()), MAX_ARTISTS)
                artists_length = min(len(dict_rec_aidx_CB.items()), len(dict_rec_aidx_CF.items()), len(dict_rec_aidx_UB.items()))
                sorted_idx_top = sorted_idx[len(sorted_idx)-artists_length:len(sorted_idx)]

                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]
            elif METHOD == "HR_CBUBCFPB_SB":     # hybrid of CF and CB, using score-based fusion (SCB)
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_UB = recommend_UB(copy_UAM, UUM, u, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_PB = recommend_PB(copy_UAM, train_aidx, MAX_ARTISTS)
                # Fuse scores given by CF and by CB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(4, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_CF.keys():
                    scores[1, aidx] = dict_rec_aidx_CF[aidx] * 0.5  # 3.72 / 12.14
                for aidx in dict_rec_aidx_UB.keys():
                    scores[2, aidx] = dict_rec_aidx_UB[aidx] * 0.2  # 3.72 / 12.14
                for aidx in dict_rec_aidx_PB.keys():
                    scores[3, aidx] = dict_rec_aidx_PB[aidx] * 0.3  # 3.72 / 12.14
                # Apply aggregation function
                scores_fused = np.max(scores, axis=0)
                # Sort and select top artists to recommend
                sorted_idx = np.argsort(scores_fused)
                # artists_length = min(len(dict_rec_aidx_CB.items()) + len(dict_rec_aidx_CF.items()), MAX_ARTISTS)
                artists_length = min(len(dict_rec_aidx_CB.items()), len(dict_rec_aidx_CF.items()), len(dict_rec_aidx_UB.items()), len(dict_rec_aidx_PB.items()))
                sorted_idx_top = sorted_idx[len(sorted_idx)-artists_length:len(sorted_idx)]

                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]

            elif METHOD == "HR_CBPB_RB":     # hybrid of CB and PB, using rank-based fusion (RB), Borda rank aggregation
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_PB = recommend_PB(copy_UAM, train_aidx, MAX_ARTISTS)
                # Fuse scores given by CB and by PB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_PB.keys():
                    scores[1, aidx] = dict_rec_aidx_PB[aidx]
                # Convert scores to ranks
                ranks = np.zeros(shape=(2, no_artists), dtype=np.int32)         # init rank matrix
                for m in range(0, scores.shape[0]):                             # for all methods to fuse
                    aidx_nz = np.nonzero(scores[m])[0]                          # identify artists with positive scores
                    scores_sorted_idx = np.argsort(scores[m,aidx_nz])           # sort artists with positive scores according to their score
                    # Insert votes (i.e., inverse ranks) for each artist and current method
                    for a in range(0, len(scores_sorted_idx)):
                        ranks[m, aidx_nz[scores_sorted_idx[a]]] = a + 1
                # Sum ranks over different approaches
                ranks_fused = np.sum(ranks, axis=0)
                # Sort and select top K_HR artists to recommend
                sorted_idx = np.argsort(ranks_fused)
                sorted_idx_top = sorted_idx[-MAX_ARTISTS:]
                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = ranks_fused[sorted_idx_top[i]]
            elif METHOD == "HR_UBCF_RB":     # hybrid of CB and PB, using rank-based fusion (RB), Borda rank aggregation
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_UB = recommend_UB(copy_UAM, UUM, u, train_aidx, K, MAX_ARTISTS)
                # Fuse scores given by CB and by PB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CF.keys():
                    scores[0, aidx] = dict_rec_aidx_CF[aidx]
                for aidx in dict_rec_aidx_UB.keys():
                    scores[1, aidx] = dict_rec_aidx_UB[aidx]
                # Convert scores to ranks
                ranks = np.zeros(shape=(2, no_artists), dtype=np.int32)         # init rank matrix
                for m in range(0, scores.shape[0]):                             # for all methods to fuse
                    aidx_nz = np.nonzero(scores[m])[0]                          # identify artists with positive scores
                    scores_sorted_idx = np.argsort(scores[m,aidx_nz])           # sort artists with positive scores according to their score
                    # Insert votes (i.e., inverse ranks) for each artist and current method
                    for a in range(0, len(scores_sorted_idx)):
                        ranks[m, aidx_nz[scores_sorted_idx[a]]] = a + 1
                # Sum ranks over different approaches
                ranks_fused = np.sum(ranks, axis=0)
                # Sort and select top K_HR artists to recommend
                sorted_idx = np.argsort(ranks_fused)
                sorted_idx_top = sorted_idx[-MAX_ARTISTS:]
                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = ranks_fused[sorted_idx_top[i]]
            elif METHOD == "HR_CBCF_RB":     # hybrid of CB and PB, using rank-based fusion (RB), Borda rank aggregation
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                # Fuse scores given by CB and by PB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CF.keys():
                    scores[0, aidx] = dict_rec_aidx_CF[aidx]
                for aidx in dict_rec_aidx_CB.keys():
                    scores[1, aidx] = dict_rec_aidx_CB[aidx]
                # Convert scores to ranks
                ranks = np.zeros(shape=(2, no_artists), dtype=np.int32)         # init rank matrix
                for m in range(0, scores.shape[0]):                             # for all methods to fuse
                    aidx_nz = np.nonzero(scores[m])[0]                          # identify artists with positive scores
                    scores_sorted_idx = np.argsort(scores[m,aidx_nz])           # sort artists with positive scores according to their score
                    # Insert votes (i.e., inverse ranks) for each artist and current method
                    for a in range(0, len(scores_sorted_idx)):
                        ranks[m, aidx_nz[scores_sorted_idx[a]]] = a + 1
                # Sum ranks over different approaches
                ranks_fused = np.sum(ranks, axis=0)
                # Sort and select top K_HR artists to recommend
                sorted_idx = np.argsort(ranks_fused)
                sorted_idx_top = sorted_idx[-MAX_ARTISTS:]
                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = ranks_fused[sorted_idx_top[i]]

            elif METHOD == "HR_CBCFPB_RB":     # hybrid of CB and PB, using rank-based fusion (RB), Borda rank aggregation
                dict_rec_aidx_CB = recommend_CB(AAM, train_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_CF = recommend_CF(copy_UAM, u, train_aidx, test_aidx, K, MAX_ARTISTS)
                dict_rec_aidx_PB = recommend_PB(copy_UAM, train_aidx, MAX_ARTISTS)
                # Fuse scores given by CB and by PB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(3, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CF.keys():
                    scores[0, aidx] = dict_rec_aidx_CF[aidx]
                for aidx in dict_rec_aidx_CB.keys():
                    scores[1, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_PB.keys():
                    scores[2, aidx] = dict_rec_aidx_PB[aidx]
                # Convert scores to ranks
                ranks = np.zeros(shape=(3, no_artists), dtype=np.int32)         # init rank matrix
                for m in range(0, scores.shape[0]):                             # for all methods to fuse
                    aidx_nz = np.nonzero(scores[m])[0]                          # identify artists with positive scores
                    scores_sorted_idx = np.argsort(scores[m,aidx_nz])           # sort artists with positive scores according to their score
                    # Insert votes (i.e., inverse ranks) for each artist and current method
                    for a in range(0, len(scores_sorted_idx)):
                        ranks[m, aidx_nz[scores_sorted_idx[a]]] = a + 1
                # Sum ranks over different approaches
                ranks_fused = np.sum(ranks, axis=0)
                # Sort and select top K_HR artists to recommend
                sorted_idx = np.argsort(ranks_fused)
                sorted_idx_top = sorted_idx[-MAX_ARTISTS:]
                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = ranks_fused[sorted_idx_top[i]]

            # Distill recommended artist indices from dictionary returned by the recommendation functions
            rec_aidx = dict_rec_aidx.keys()

            # print "Recommended items: ", len(rec_aidx)

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
            # print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    t1 = time()
    pastedTime = t1 - t0

    # Output mean average precision and recall
    print str(MAX_ARTISTS) + "/" + str(K) + ": MAP %.2f" % avg_prec + " MAR %.2f" % avg_rec + " Time %.2f" % pastedTime

    with open(filename, "a") as myfile:
        myfile.write(str(K) + "\t" + str(MAX_ARTISTS) + "\t" + "%.2f" % avg_prec + "\t" + "%.2f" % avg_rec + "\t" + "%.2f" % pastedTime + "\n")

    print "FINISHED"


# Main program
if __name__ == '__main__':

    processes = []

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
    # Load AAM
    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)
    # Load UUM
    UUM = np.loadtxt(UUM_FILE, delimiter='\t', dtype=np.float32)

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]

    
    no_artists_rec = [1,2,3,5,8,13,21,34,55,89,144]
    
    # RB
    # CF _k, _artists, _artists_k
    # CB _k, _artists, _artists_k
    # PB _k, _artists, _artists_k
    # UB _k, _artists, _artists_k
    # UBCF _k, _artists, _artists_k

    # HR_UBCF_SB _k, _artists, _artists_k
    # HR_CBCF_SB _k, _artists, _artists_k
    # HR_CBPB_SB _k, _artists, _artists_k
    # HR_CBUBCF_SB _k, _artists, _artists_k
    # HR_CBUBCFPB_SB _k, _artists, _artists_k

    # HR_UBCF_RB _k, _artists, _artists_k
    # HR_CBCF_RB _k, _artists, _artists_k
    # HR_CBPB_RB _k, _artists, _artists_k

    K = 10           # for CB: number of nearest neighbors to consider for each artist in seed user's training set

    try:
        for METHOD in ["RB", "RBU", "CF", "UB", "UBCF", "CB", "PB", "HR_CBCF_SB", "HR_UBCF_SB", "HR_CBPB_SB", "HR_CBUBCF_SB", "HR_CBUBCFPB_SB", "HR_CBPB_RB", "HR_UBCF_RB", "HR_CBCF_RB", "HR_CBCFPB_RB"]:
            MAX_ARTISTS = 0
            foldername = "./results/"
            if not os.path.exists(foldername):
                os.makedirs(foldername)

            filename = foldername + str(METHOD) + "_k.txt"
            with open(filename, "w") as myfile:
                myfile.write("K" + "\tArtists" + "\tPrec" + "\tRec" + "\tel. Time" + "\n")
            for MAX_ARTISTS in no_artists_rec:

                print "Starting " + METHOD + " with " + str(MAX_ARTISTS) + " artists"

                p = Process(target=run, args=(artists, users, UAM, UUM, AAM, no_users, no_artists, METHOD, K, MAX_ARTISTS))
                #sp = Process(target=run, args=(artists, users, UAM, no_users, no_artists, METHOD, K, MAX_ARTISTS))
                processes += [p]
                p.start()
                if len(processes) % 3 == 0:
                    for x in processes:
                        x.join()
                    processes = []
    except Exception,e:
        print str(e)
