#!/usr/bin/env python
# Preprocessing code to create a file with all necessary data

import json
import sys
sys.path.append("/home/xbt504/nlp-project/src/")

from langdetect import detect
from pickle import dump
from util import getText, getCharMapping, getNameMapping, getFullName, getCoords

# Preprocessing parameters
sequence_length = 10
test_set_perc = 10.0
filename = "sequences.txt"
file_da = '../data/tweets_collected_da.jsonl'
file_sv = '../data/tweets_collected_sv.jsonl'
file_no = '../data/tweets_collected_no.jsonl'

def filterTweets(data):
    to_remove = []
    for index, item in enumerate(data):
        tweet = json.loads(item)
        timestamp = tweet['created_at']
        # Check if it's likely to be an hourly weather tweet
        if (timestamp[14] == '0' and
            timestamp[15] == '0' and
            timestamp[17] == '0' and
            timestamp[18] < '5'):
            to_remove.append(index)
        else:
            text = tweet['text']
            if (detect(text) == 'en'):
                # Check if the tweet is in english
                to_remove.append(index)

    for index in to_remove[::-1]:
        data.pop(index)
    return data

da = open(file_da, 'r').read().splitlines()
sv = open(file_sv, 'r').read().splitlines()
no = open(file_no, 'r').read().splitlines()

tweets = (da + sv + no)[:100]
print "Loaded %d tweets" % len(tweets)

tweets = filterTweets(tweets)

# Remove duplicates
print "%d tweets before removing duplicates" % len(tweets)
tweets = list(set(tweets))
print "%d tweets left after preprocessing" % len(tweets)

# Create mappings and write to file
char_mapping = getCharMapping(tweets)
name_mapping = getNameMapping(tweets)
print "vocab size: %d" % len(char_mapping)
dump(name_mapping, open('name_mapping.pkl', 'wb'))
dump(char_mapping, open('char_mapping.pkl', 'wb'))
outfile = open(filename, "a")

for tweet in tweets:
    full_name = getFullName(tweet)
    text = getText(tweet)
    coords = getCoords(tweet)
    lat = coords[0]
    lon = coords[1]
    for i in range(sequence_length, len(text)):
        seq = text[i - sequence_length:i + 1]
        encoded_seq = [char_mapping[char] for char in seq]
        encoded_name = name_mapping[full_name]
        # Encode strings
        string = '%s %d %f %f\n' % ('-'.join(str(n) for n in encoded_seq), encoded_name, lat, lon)
        outfile.write(string)

