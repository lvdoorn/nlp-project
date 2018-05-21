#!/usr/bin/env python
#
# Preprocessing code to create files with all necessary data
# Output files are created in the output_dir directory

import json
import collections
import os.path
import sys
import random
from collections import Counter

sys.path.append("/home/xbt504/nlp-project/src/")

from langdetect import detect
from pickle import dump
from pickle import load
from keras.preprocessing.sequence import pad_sequences 
from tweet import Tweet

# Preprocessing parameters
max_chars = 250
test_set_ratio = 0.1

# Output filenames
output_dir = '../out'
output_filename = "%s/sequences.txt" % output_dir
noDuplicatesFilename = '%s/no_duplicates.txt' % output_dir
trainSetFilename = '%s/train_set.txt' % output_dir
testSetNormalFilename = '%s/test_set_normal.txt' % output_dir
testSetLocationFilename = '%s/test_set_location.txt' % output_dir
char_mapping_filename = "%s/char_mapping.pkl" % output_dir
name_mapping_filename = "%s/name_mapping.pkl" % output_dir
country_filename = "%s/countries.txt" % output_dir

# Input filenames
data_dir = '../raw_data'
file_da = '%s/tweets_collected_da.jsonl' % data_dir
file_sv = '%s/tweets_collected_sv.jsonl' % data_dir
file_no = '%s/tweets_collected_no.jsonl' % data_dir

# Reads input files, removes duplicate lines and writes to a single file
def removeDuplicates():
    da = open(file_da, 'r').read().splitlines()
    sv = open(file_sv, 'r').read().splitlines()
    no = open(file_no, 'r').read().splitlines()
    tweets = (da + sv + no)
    original_length = len(tweets)
    print "Loaded %d tweets from file" % original_length
    print "Removing duplicates ..."
    tweets = list(set(tweets))

    # Remove tweets without location data
    remainingTweets = []
    for tweet in tweets:
        try:
            js = json.loads(tweet)
            if (not ('place' in js)) or js['place'] == None:
                continue
            elif (not ('full_name' in js['place'])):
                continue
            elif (not ('geo' in js)) or js['geo'] == None:
                continue
            elif (not ('coordinates' in js['geo'])):
                continue
            remainingTweets.append(tweet)
        except ValueError:
            pass
    tweets = remainingTweets
    print "%d duplicates removed" % (original_length - len(tweets))
    open(noDuplicatesFilename, 'w').write('\n'.join(tweets))
    print "Wrote %d tweets to file" % len(tweets)
    
# Reads all tweets and splits them into a train and test set, which are written to file
def createTrainTestSets():
    tweets = open(noDuplicatesFilename, 'r').read().splitlines()
    name_mapping = loadNameMapping()
    holdoutLocations = [u'Frederiksberg, Danmark', u'T\xe5rnby, Danmark', u'Kolding, Danmark', u'T\xe4by, Sverige', u'Kungsbacka, Sverige', u'Kristianstad, Sverige', u'Bod\xf8, Norge', u'Kvinnherad, Norge', u'Ullensaker, Norge']
    testSetLocation = []
    rest = []
    for tweet in tweets:
        if stringToTweet(tweet).getFullName() in holdoutLocations:
            testSetLocation.append(tweet)
        else:
            rest.append(tweet)
    tweets = rest
    testIndex = int(round(len(tweets) * (1 - test_set_ratio)))
    random.seed(1)
    random.shuffle(tweets)
    trainSet = tweets[:testIndex]
    testSet = tweets[testIndex:]
    open(trainSetFilename, 'w').write('\n'.join(trainSet))
    open(testSetNormalFilename, 'w').write('\n'.join(testSet))
    open(testSetLocationFilename, 'w').write('\n'.join(testSetLocation))
    print "Wrote %d tweets to train set" % len(trainSet)
    print "Wrote %d tweets to normal test set" % len(testSet)
    print "Wrote %d tweets to location test set" % len(testSetLocation)

def createTrainTestSetsLocation():
    tweets = open(noDuplicatesFilename, 'r').read().splitlines()
    name_mapping = loadNameMapping()
    holdoutLocations = [u'Frederiksberg, Danmark', u'T\xe5rnby, Danmark', u'Kolding, Danmark', u'T\xe4by, Sverige', u'Kungsbacka, Sverige', u'Kristianstad, Sverige', u'Bod\xf8, Norge', u'Kvinnherad, Norge', u'Ullensaker, Norge']
    tweets = loadTweets(noDuplicatesFilename)
    holdout = []
    rest = []
    for tweet in tweets:
        name = tweet.getFullName()
        if name in holdoutLocations:
            holdout.append(tweet)
        else:
            rest.append(tweet)
    open(trainSetLocationFilename, 'w').write('\n'.join(trainSet))
    open(testSetLocationFilename, 'w').write('\n'.join(testSet))
    print "Wrote %d tweets to train set" % len(trainSet)
    print "Wrote %d tweets to test set" % len(testSet)

# Reads filename, and returns a list of Tweet objects
def loadTweets(filename):
    tweets = open(filename, 'r').read().splitlines()
    print "Loading %d tweets from %s ..." % (len(tweets), filename)
    tweetObjects = []
    for tweet in tweets:
        try:
            js = json.loads(tweet)
            if (not ('place' in js)) or js['place'] == None:
                continue
            elif (not ('full_name' in js['place'])):
                continue
            elif (not ('geo' in js)) or js['geo'] == None:
                continue
            elif (not ('coordinates' in js['geo'])):
                continue
            coords = js['geo']['coordinates']
            place = js['place']
            tweetObject = Tweet(js['text'], place['full_name'], coords[0], coords[1], place['country'], js['created_at'])
            tweetObjects.append(tweetObject)
        except ValueError:
            pass
    print "Loaded %d tweets" % len(tweetObjects)
    return tweetObjects

def stringToTweet(string):
    js = json.loads(string)
    coords = js['geo']['coordinates']
    place = js['place']
    return Tweet(js['text'], place['full_name'], coords[0], coords[1], place['country'], js['created_at'])

def getCharMapping(tweets):
    text = map(lambda x: x.getText(), tweets)
    allChars = [c for s in text for c in s]
    x = collections.Counter(allChars)
    chars_used = x.most_common()[:max_chars]
    charset = map(lambda x: x[0], chars_used)
    # Add padding, start, end and unknown characters
    mapping = dict((c, i) for i, c in enumerate(charset + ['<s>', '</s>', '<pad>', '<unknown>', '<unknown_test>']))
    dump(mapping, open(char_mapping_filename, 'wb'))
    return mapping

def getNameMapping(tweets):
    allNames = []
    for tweet in tweets:
        allNames.append(tweet.getFullName())
    names = sorted(list(set(allNames)))
    mapping = dict((c, i) for i, c in enumerate(names + ['<pad>']))
    dump(mapping, open(name_mapping_filename, 'wb'))
    return mapping

def getCharTrain(c, mapping):
    unknown_value = mapping['<unknown>']
    return mapping.get(c, unknown_value)

def getCharTest(c, mapping):
    unknown_value = 254 # mapping['<unknown_test>']
    return mapping.get(c, unknown_value)

def loadCharMapping():
    return load(open(char_mapping_filename, 'rb'))

def loadNameMapping():
    return load(open(name_mapping_filename, 'rb'))
