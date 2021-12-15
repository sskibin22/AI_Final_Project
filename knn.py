import random
from typing import DefaultDict, Text
import numpy as np
import timeit
import math
import statistics

# globals
FACE_BLOCK_LEN = 10


def viewAsBlocks(arr, BSZ):
    # arr is input array, BSZ is block-size
    m, n = arr.shape
    M, N = BSZ
    return arr.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M, N)


def computeFeature(block):
    count = 0
    totalCount = 0
    for row in block:
        for element in row:
            totalCount += 1
            if element == 1:
                count += 1

    return count / totalCount


def matrixify(arr, mode):
    width = 0
    if mode == 1:
        width = 28
    else:
        width = 60
    numberImg = []
    for line in arr:
        for i in range(width):
            if not line[i].strip():
                numberImg.append(0)
            else:
                numberImg.append(1)

    matrix = np.array(numberImg)
    if mode == 0:
        matrix = matrix.reshape(70, 60)

    return matrix


def euclideanDistance(img1, img2, mode):
    # if mode == 0 extract features
    if mode == 0:
        mat1 = viewAsBlocks(img1, (FACE_BLOCK_LEN, FACE_BLOCK_LEN))
        mat2 = viewAsBlocks(img2, (FACE_BLOCK_LEN, FACE_BLOCK_LEN))
        features1 = [computeFeature(block) for block in mat1]
        features2 = [computeFeature(block) for block in mat2]
        currSum = 0

        for f1, f2 in zip(features1, features2):
            squaredDifference = (f1 - f2) ** 2
            currSum += squaredDifference
        return math.sqrt(currSum)
    else:
        return sum((img1 - img2) ** 2)


def majorityLabel(labels):
    counterDict = DefaultDict(int)
    for label in labels:
        counterDict[label] += 1

    majorityLabel = max(counterDict.values())
    for key, value in counterDict.items():
        if value == majorityLabel:
            return key


def predict(k, trainingImages, trainingLabels, testImage, mode):
    distances = [
        (euclideanDistance(testImage, image, mode), label)
        for (image, label) in zip(trainingImages, trainingLabels)
    ]

    # sort
    sortedDistances = sorted(distances, key=lambda tup: tup[0])
    # for tup in by_distances:
    # print(tup)

    # get k closest labels
    kClosest = [label for (_, label) in sortedDistances[:k]]

    res = majorityLabel(kClosest)
    # print(res)
    return res


def getImages(lines, labels, mode):
    images = []
    intlabels = []
    width = 0

    if mode == 1:
        length = 28
    else:
        length = 70

    # create matrix-label tuples and store in list
    left = 0
    right = length
    count = 0

    while right <= len(lines):
        # add matrix label tuples
        images.append((matrixify(lines[left:right], mode)))
        intlabels.append(int(labels[count]))
        left = right
        right = right + length
        count += 1

    return images, intlabels


def main(mode):
    start = timeit.default_timer()

    testImagesPath = ""
    testLabelsPath = ""
    trainImagesPath = ""
    trainLabelsPath = ""

    if mode == 1:  # digits
        print("CLASSIFYING DIGITS")
        testImagesPath = "digitdata/testimages"
        testLabelsPath = "digitdata/testlabels"
        trainImagesPath = "digitdata/trainingimages"
        trainLabelsPath = "digitdata/traininglabels"
    else:  # face
        print("CLASSIFYING FACES")
        testImagesPath = "facedata/facedatatest"
        testLabelsPath = "facedata/facedatatestlabels"
        trainImagesPath = "facedata/facedatatrain"
        trainLabelsPath = "facedata/facedatatrainlabels"

    # read the testing numbers into list
    testFile = open("{}".format(testImagesPath), "r")
    testLines = testFile.readlines()
    testFile.close()

    # read testing labels for digits into list
    testLabelFile = open("{}".format(testLabelsPath), "r")
    testLabelLines = testLabelFile.readlines()
    testLabelFile.close()

    # read the training numbers into list
    digitFile = open("{}".format(trainImagesPath), "r")
    digitLines = digitFile.readlines()
    digitFile.close()

    # read training labels for digits into list
    labelFile = open("{}".format(trainLabelsPath), "r")
    labelLines = labelFile.readlines()
    labelFile.close()

    testingImages, testingLabels = getImages(testLines, testLabelLines, mode)
    trainingImages, trainingLabels = getImages(digitLines, labelLines, mode)

    trainDataAmount = int(round(0.1 * len(trainingImages)))
    accuracies = []
    print(len(trainingImages[:trainDataAmount]))
    # randomize testing images
    randomizedImages = [tup for tup in zip(testingImages, testingLabels)]
    randomizedImages = random.sample(randomizedImages, len(randomizedImages))

    print(len(randomizedImages))

    numCorrect = 0
    i = 0
    for j in range(5):
        for testImage, testLabel in randomizedImages[:500]:
            prediction = predict(
                10,
                trainingImages[:trainDataAmount],
                trainingLabels[:trainDataAmount],
                testImage,
                mode,
            )
            if prediction == testLabel:
                numCorrect += 1
            i += 1

        accuracies.append(numCorrect / 500)
        print(accuracies[0])

    # print(sum(accuracies) / len(accuracies))
    print(statistics.variance(accuracies))
    stop = timeit.default_timer()
    print(stop - start)


main(1)
