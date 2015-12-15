# Post-process the crawled music context data, extract term weights, and compute cosine similarities.
__author__ = "Eschbacher - Kratzer - Scharfetter"

import os
import numpy as np
import scipy.spatial.distance as scidist      # import distance computation module from scipy package
import urllib
import helpers
import csv
import sys

MIN_TERM_DF = 2

# Stop words used by Google
STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

def readFile(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter='\t')  # create reader
        headers = reader.next()  # skip header
        for row in reader:
            item = row[1]
            data.append(item)
    f.close()
    return data


# Main program
if __name__ == "__main__":

    pathToArtistInfo = str(sys.argv[1])
    pathToUniqueArtists = str(sys.argv[2])
    pathToOverall = str(sys.argv[3])

    artists = helpers.readFile(pathToUniqueArtists)

    htmlContents = {}
    termsDF = {}

    # for all artists
    for i in range(0, len(artists)):

        htmlFn = pathToArtistInfo + "/" + str(i) + ".html"

        if os.path.exists(htmlFn):

            htmlContent = open(htmlFn, "r").read()

            contentPure = helpers.removeHTMLMarkup(htmlContent)
            contentCasefolded = contentPure.lower()

            # Tokenize stripped content at white space characters
            tokens = contentCasefolded.split()

            # Remove all tokens containing non-alphanumeric characters; using a simple lambda function (i.e., anonymous function, can be used as parameter to other function)
            tokensFiltered = filter(lambda t: t.isalnum(), tokens)

            # Remove words in the stop word list
            tokensFilteredWithStopwords = filter(lambda t: t not in STOP_WORDS, tokensFiltered)

            htmlContents[i] = tokensFilteredWithStopwords
            print "File " + htmlFn + " --- total tokens: " + str(len(tokens)) + "; after filtering and stopping: " + str(len(tokensFilteredWithStopwords))
        else:
            print "Target file " + htmlFn + " does not exist!"


    # Start computing term weights, in particular, document frequencies and term frequencies.

    # Iterate over all (key, value) tuples from dictionary just created to determine document frequency (DF) of all terms
    for (aid, terms) in htmlContents.items():
        # convert list of terms to set of terms ("uniquify" words for each artist/document)
        for t in set(terms):
            # update number of artists/documents in which current term t occurs
            if not termsDF.has_key(t):
                termsDF[t] = 1
            else:
                termsDF[t] += 1

    if MIN_TERM_DF > 1:
        tempTermsDF = {}
        for key, val in termsDF.items():
            if val >= MIN_TERM_DF:
                tempTermsDF[key] = val

        termsDF = tempTermsDF

    # Compute number of artists/documents and terms
    numberArtists = len(htmlContents.items())
    numberTerms = len(termsDF)
    print "Number of artists in corpus: " + str(numberArtists)
    print "Number of terms in corpus: " + str(numberTerms)

    # You may want (or need) to perform some kind of dimensionality reduction here, e.g., filtering all terms
    # with a very small document frequency.
    # ...

    termList = []

    # Dictionary is unordered, so we store all terms in a list to fix their order, before computing the TF-IDF matrix
    for t in termsDF.keys():
        termList.append(t)

    termLookup = {}
    for tIdx in range(0, len(termList)):
        termLookup[termList[tIdx]] = tIdx

    # Create IDF vector using logarithmic IDF formulation
    idf = np.zeros(numberTerms, dtype=np.float32)
    for i in range(0, numberTerms):
        idf[i] = np.log(numberArtists / termsDF[termList[i]])

    # Initialize matrix to hold term frequencies (and eventually TF-IDF weights) for all artists for which we fetched HTML content
    tfidf = np.zeros(shape=(numberArtists, numberTerms), dtype=np.float32)

    # Iterate over all (artist, terms) tuples to determine all term frequencies TF_{artist,term}

    for aIdx, terms in htmlContents.items():
        print "Computing term weights for artist " + str(aIdx)
        for t in terms:
            if (termLookup.has_key(t)):
                tIdx = termLookup[t]
                tfidf[aIdx-1, tIdx] += 1

    # Replace TF values in tfidf by TF-IDF values:
    # copy and reshape IDF vector and point-wise multiply it with the TF values
    # tfidf = np.log1p(tfidf) * np.tile(idf, no_artists).reshape(no_artists, no_terms)

    tfidf = np.log1p(tfidf)
    for i in range(0, numberArtists):
        tfidf[i,:] *= idf

    # Storing TF-IDF weights and term list
    output = pathToOverall + "tfidf.csv"
    print "Saving TF-IDF matrix to " + output + "."
    np.savetxt(output, tfidf, fmt="%0.6f", delimiter="\t", newline="\n")

    output = pathToOverall + "terms.csv"
    print "Saving term list to " + output + "."
    with open(output, "w") as f:
        for t in termList:
            f.write(t + "\n")

    # Computing cosine similarities and store them
    # print "Computing cosine similarities between artists."
    # Initialize similarity matrix
    sims = np.zeros(shape=(numberArtists, numberArtists), dtype=np.float32)
    # Compute pairwise similarities between artists
    for i in range(0, numberArtists):
        print "Computing similarities for artist " + str(i)
        for j in range(i, numberArtists):
            cossim = 1.0 - scidist.cosine(tfidf[i], tfidf[j])
            # If either TF-IDF vector (of i or j) only contains zeros, cosine similarity is not defined (NaN: not a number).
            # In this case, similarity between i and j is set to zero (or left at zero, in our case).
            if not np.isnan(cossim):
                sims[i,j] = cossim
                sims[j,i] = cossim

    output = pathToOverall + "aam.csv"
    print "Saving cosine similarities to " + output + "."
    np.savetxt(output, sims, fmt="%0.6f", delimiter="\t", newline="\n")
