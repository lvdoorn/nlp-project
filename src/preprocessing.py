import json
from langdetect import detect

def removeAutomated(tweetList):
    to_remove = []
    data = tweetList
    for index, item in enumerate(data):
        tweet = json.loads(item)
        timestamp = tweet['created_at']
        # Check if it's likely to be an hourly weather tweet
        if (timestamp[14] == '0' and timestamp[15] == '0' and timestamp[17] == '0' and timestamp[18] < '5'):
            to_remove.append(index)

    print 'Removing %d / %d automated tweets' % (len(to_remove), len(tweetList))
    for index in to_remove[::-1]:
        data.pop(index)
    print 'New data size: %d' % len(data)
    print ''
    return data

def removeEnglish(tweetList):
    to_remove = []
    data = tweetList
    for index, item in enumerate(data):
        tweet = json.loads(item)
        text = tweet['text']
        # Check if tweet is in English
        if (detect(text) == 'en'):
            to_remove.append(index)
    print 'Removing %d / %d English tweets' % (len(to_remove), len(tweetList))
    for index in to_remove[::-1]:
        data.pop(index)
    print 'New data size: %d' % len(data)
    print ''   
    return data

