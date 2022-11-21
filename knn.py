from csv import reader
import random
import math
import operator


def load_dataset(filename, split, training_set=[], test_set=[]):
    with open(filename) as f:
        lines = reader(f)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(6):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((int(instance1[x]) - int(instance2[x])), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance)
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


training_set = []
test_set = []
split = 0.66
load_dataset("hayes_roth.data", split, training_set, test_set)

predictions = []
k = 3
for x in range(len(test_set)):
    neighbors = get_neighbors(training_set, test_set[x], k)
    result = get_response(neighbors)
    predictions.append(result)
accuracy = get_accuracy(test_set, predictions)
print("Accuracy: ", accuracy, "%")