def getBallDistributionData(dataMatrixInt, ballColourFreqArray):

    print(dataMatrixInt)

    for row in dataMatrixInt:
        for e in row:
            ballColourFreqArray[e] += 1

    return ballColourFreqArray
