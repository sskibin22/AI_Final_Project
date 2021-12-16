import random
from typing import DefaultDict, Text
import numpy as np
import timeit
import math
import statistics

# globals
FACE_BLOCK_LEN = 10
DIGIT_BLOCK_LEN = 2


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


def euclideanDistance(img1, mode, features):
    # if mode == 0 extract features
    # if mode == 0:
    mat1 = None
    if mode == 0:
        mat1 = viewAsBlocks(img1, (FACE_BLOCK_LEN, FACE_BLOCK_LEN))
    else:
        mat1 = np.array(img1)
        mat1 = mat1.reshape(28, 28)
        mat1 = viewAsBlocks(mat1, (DIGIT_BLOCK_LEN, DIGIT_BLOCK_LEN))
    features1 = [computeFeature(block) for block in mat1]
    currSum = 0

    for f1, f2 in zip(features1, features):
        squaredDifference = (f1 - f2) ** 2
        currSum += squaredDifference
    return math.sqrt(currSum)
    # else:
    #  return sum((img1 - img2) ** 2)


def majorityLabel(labels):
    counterDict = DefaultDict(int)
    for label in labels:
        counterDict[label] += 1

    majorityLabel = max(counterDict.values())
    for key, value in counterDict.items():
        if value == majorityLabel:
            return key


def predict(k, trainingImages, trainingLabels, trainingFeatures, testImage, mode):
    distances = [
        (euclideanDistance(testImage, mode, features), label)
        for (features, label) in zip(trainingFeatures, trainingLabels)
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


def extractFeatures(images, mode):
    blocksList = None
    if mode == 0:
        blocksList = [
            viewAsBlocks(image, (FACE_BLOCK_LEN, FACE_BLOCK_LEN)) for image in images
        ]
    else:
        matrices = []
        for image in images:
            matrix = np.array(image)
            matrix = matrix.reshape(28, 28)
            matrices.append(matrix)
        blocksList = [
            viewAsBlocks(matrix, (DIGIT_BLOCK_LEN, DIGIT_BLOCK_LEN))
            for matrix in matrices
        ]
    featuresList = []
    for blocks in blocksList:
        features = [computeFeature(block) for block in blocks]
        featuresList.append(features)

    return featuresList


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

    print(len(trainingImages))
    print(len(testingImages))

    trainingImagesFeatures = extractFeatures(trainingImages, mode)

    trainDataAmount = int(round(1 * len(trainingImages)))
    accuracies = []
    print(len(trainingImages[:trainDataAmount]))
    # randomize testing images
    randomizedImages = [tup for tup in zip(testingImages, testingLabels)]
    randomizedImages = random.sample(randomizedImages, len(randomizedImages))

    print(len(randomizedImages))

    numCorrect = 0
    i = 0

    for testImage, testLabel in randomizedImages:
        print(i)
        prediction = predict(
            10,
            trainingImages[:trainDataAmount],
            trainingLabels[:trainDataAmount],
            trainingImagesFeatures[:trainDataAmount],
            testImage,
            mode,
        )
        if prediction == testLabel:
            print("correct")
            numCorrect += 1
        else:
            print("incorrect")

        i += 1

    print(numCorrect / len(randomizedImages))

    stop = timeit.default_timer()
    print(stop - start)


main(1)
