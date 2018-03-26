#!/usr/bin/env python
# Preprocessing code to create a file with all necessary data

import json
import sys
sys.path.append("/home/xbt504/nlp-project/src/")

from langdetect import detect
from pickle import dump
from util import getText, getCharMapping, getNameMapping, getFullName, getCoords
from keras.preprocessing.sequence import pad_sequences 

# Preprocessing parameters
sequence_length = 280 # maximum tweet length
test_set_perc = 10.0
output_filename = "sequences.txt"
file_da = '../data/tweets_collected_da.jsonl'
file_sv = '../data/tweets_collected_sv.jsonl'
file_no = '../data/tweets_collected_no.jsonl'

def filterTweets(data):
    to_remove = []
    for index, item in enumerate(data):
        print item
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

tweets = (da + sv + no)[:200]
original_length = len(tweets)
print "Loaded %d tweets from file" % original_length
print "Removing duplicates..."
tweets = list(set(tweets))
print "%d duplicates removed" % (original_length - len(tweets))
print "Filtering %d tweets based on language and automation..." % len(tweets)
tweets = filterTweets(tweets)
print "%d tweets left after preprocessing" % len(tweets)


# Create mappings and write to file
# Padding and start and end marks in mapping are handled in util
char_mapping = getCharMapping(tweets)
name_mapping = getNameMapping(tweets)
print "vocab size: %d" % len(char_mapping)
dump(name_mapping, open('name_mapping.pkl', 'wb'))
dump(char_mapping, open('char_mapping.pkl', 'wb'))

# Pad sequences up to maximum length, add start and end marks and write to file (as json)
outfile = open(output_filename, "a")
sequences = []
for tweet in tweets:
    full_name = getFullName(tweet)
    text = getText(tweet)
    coords = getCoords(tweet)
    lat = coords[0]
    lon = coords[1]
    sequence = [char_mapping[char] for char in text]
    sequence = [char_mapping['<s>']] + sequence + [char_mapping['</s>']]
    sequence = pad_sequences([sequence], maxlen=sequence_length+2, value=char_mapping['<pad>'], padding='post')[0]
    print sequence
    encoded_name = name_mapping[full_name]
    # Encode strings
    string = '%s %d %f %f\n' % (','.join(str(n) for n in sequence), encoded_name, lat, lon)
    print string
    outfile.write(string)

