import random
def splitData(data, locationBased, percentage):
    random.shuffle(data) 
    testSet = []
    locations = map(lambda x: x['name'], data)
    for loc in locations:
        allLoc = filter(lambda x: x['name'] == loc, data)
        testSet.extend(allLoc)
        if (len(testSet) / float(len(data)) > percentage):
            return testSet
