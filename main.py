# from scipy.stats import itemfreq
import constants as const
import autoGui as aGUI
import utils as utils
import isPathExisting as isPathExisting
import ballDistData

from matplotlib import pyplot as plt
import pyscreenshot as ImageGrab
import cv2
import math
import numpy as np
import random
import time
import json as json
from sklearn.cluster import KMeans

# np.set_printoptions(threshold=np.nan)


def areColorSimilar(color1, color2):

    one = abs(color1[0] - color2[0])
    two = abs(color1[1] - color2[1])
    three = abs(color1[2] - color2[2])

    if (one < const.COLOR_SIMILAR_FACTOR and two < const.COLOR_SIMILAR_FACTOR and three < const.COLOR_SIMILAR_FACTOR):
        return True
    else:
        return False


def getBallColorName(color):

    if areColorSimilar(color, const.redBall):
        return 1
    elif areColorSimilar(color, const.blueBall):
        return 2
    elif areColorSimilar(color, const.orangeBall):
        return 3
    elif areColorSimilar(color, const.yellowBall):
        return 4
    elif areColorSimilar(color, const.purpleBall):
        return 5
    elif areColorSimilar(color, const.greenBall):
        return 6
    elif areColorSimilar(color, const.cyanBall):
        return 7
    else:
        return 0


def getDominantColor(img2):

    arr = np.float32(img2)
    pixels = arr.reshape((-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01)
    flags = cv2.KMEANS_RANDOM_CENTERS

    rev, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 5, flags)
    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img2.shape)
    # dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

    # Assuming 'labels' is the array containing cluster labels from k-means clustering
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Find the index of the label with the maximum count
    dominant_label_index = np.argmax(label_counts)

    # Use the dominant label index to get the corresponding color from 'palette'
    dominant_color = palette[unique_labels[dominant_label_index]]

    return dominant_color


def convertRowToBinary(row):
    binaryRow = np.zeros(const.GRID_ROW, dtype=int)
    binaryRowCnt = 0
    for i in range(0, const.GRID_ROW):
        if row[i] != -1:
            binaryRow[binaryRowCnt] = 1
        else:
            binaryRow[binaryRowCnt] = 0
        binaryRowCnt += 1
    return binaryRow


def getDataMatrix():

    x, y = const.TOPLEFT_X, const.TOPLEFT_Y
    width, height = const.BOTTOM_RIGHT_X - x, const.BOTTOM_RIGHT_Y-y

    # Capture the screen or a specific region
    image = np.array(ImageGrab.grab(bbox=(x, y, x + width, y + height)))
    # image = np.array(ImageGrab.grab(bbox=(25, 88, 325, 390)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # cv2.imshow('Captured Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rowCnt = 1
    a = 1
    b = const.CURSOR_STEP_SIZE

    dataMatrixStr = np.zeros([const.GRID_ROW, const.GRID_COL], dtype=str)
    dataMatrixInt = np.zeros([const.GRID_ROW, const.GRID_COL], dtype=int)
    binaryMatrix = np.zeros([const.GRID_ROW, const.GRID_COL], dtype=int)

    row = np.zeros(const.GRID_ROW, dtype=str)
    rowInt = np.zeros(const.GRID_ROW, dtype=int)

    while (rowCnt <= const.GRID_ROW):

        rowImage = image[a:b, 1:const.CURSOR_STEP_SIZE*const.GRID_COL]

        # cv2.imshow('Captured Image', rowImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cellCnt = 1
        c = 1
        d = const.CURSOR_STEP_SIZE

        while (cellCnt <= const.GRID_COL):

            cellImage = rowImage[1:const.CURSOR_STEP_SIZE, c:d]

            dominant_color = getDominantColor(cellImage)

            ballColorCode = getBallColorName(dominant_color)-1

            row[cellCnt-1] = const.ballColorShortCodeArray[ballColorCode]
            rowInt[cellCnt-1] = ballColorCode

            c = d
            d = c + const.CURSOR_STEP_SIZE
            cellCnt += 1

        dataMatrixStr[rowCnt-1] = row
        dataMatrixInt[rowCnt-1] = rowInt

        binaryMatrix[rowCnt-1] = convertRowToBinary(rowInt)

        # print("rowInt :: "+str(rowInt))

        # print("binaryMatrix :: " +
        #       str(binaryMatrix[rowCnt-1])+"\t rowCnt :: "+str(rowCnt))

        a = b
        b = b + const.CURSOR_STEP_SIZE
        rowCnt = rowCnt + 1

    return dataMatrixInt, binaryMatrix,


def getEmptyAndOccupiedCells(dataMatrix):

    emptyArray = np.zeros(const.TOTAL_GRID_CELLS, dtype=utils.cellDT)
    occupiedArray = np.zeros(const.TOTAL_GRID_CELLS, dtype=utils.cellDT)
    occupiedArrayCnt = 0
    emptyArrayCnt = 0

    for i in range(0, const.GRID_ROW):
        for j in range(0, const.GRID_COL):
            if dataMatrix[i][j] != -1:
                occupiedArray[occupiedArrayCnt][0] = i
                occupiedArray[occupiedArrayCnt][1] = j
                occupiedArrayCnt += 1
            else:
                emptyArray[emptyArrayCnt][0] = i
                emptyArray[emptyArrayCnt][1] = j
                emptyArrayCnt += 1

    return occupiedArray, emptyArray


def getNumberOfElementsInArray(array):

    cnt = 0
    for row in array:
        if row[0] == 0 and row[1] == 0:
            cnt += 1

    return cnt


def getColorCount(dataMatrixInt):

    red = 0
    blue = 0
    orange = 0
    yellow = 0
    purple = 0
    green = 0
    cyan = 0
    noBall = 0

    for i in range(0, const.GRID_ROW):
        for j in range(0, const.GRID_COL):
            if dataMatrixInt[i][j] == 1:
                red += 1
            elif dataMatrixInt[i][j] == 2:
                blue += 1
            elif dataMatrixInt[i][j] == 3:
                orange += 1
            elif dataMatrixInt[i][j] == 4:
                yellow += 1
            elif dataMatrixInt[i][j] == 5:
                purple += 1
            elif dataMatrixInt[i][j] == 6:
                green += 1
            elif dataMatrixInt[i][j] == 7:
                cyan += 1
            elif dataMatrixInt[i][j] == 0:
                noBall += 1

    arr = np.array([red, blue, orange, yellow, purple, green, cyan, noBall])

    return arr, (red+blue+orange+yellow+purple+green+cyan)


def getLocationForCode(code, dataMatrixInt):

    locations = np.full(const.TOTAL_GRID_CELLS, -1, dtype=utils.cellDT)
    locationsCnt = 0
    for i in range(0, const.GRID_ROW):
        for j in range(0, const.GRID_COL):
            if dataMatrixInt[i][j] == code:
                locations[locationsCnt][0] = i
                locations[locationsCnt][1] = j
                locationsCnt += 1

    return locations, locationsCnt


def getemptyCellsBeforeAfter(row, code, cell):

    before = 0
    after = 0

    c = cell

    one = row[0:c]
    two = row[c+1:]

    if one.size == 0:
        before = 0
    else:
        for e in one:
            if e == 0:
                before += 1

    if two.size == 0:
        after = 0
    else:
        for e in two:
            if e == 0:
                after += 1

    return before, after


def getEmptyCellCount(cell, dataMatrixInt):

    top = 0
    bottom = 0
    right = 0
    left = 0

    row = dataMatrixInt[cell[0], :]
    col = dataMatrixInt[:, cell[1]]
    code = dataMatrixInt[cell[0]][cell[1]]

    left, right = getemptyCellsBeforeAfter(row, code, cell[1])
    top, bottom = getemptyCellsBeforeAfter(col, code, cell[0])
    countRow = np.count_nonzero(row == code)
    countCol = np.count_nonzero(col == code)

    return top, bottom, right, left, countRow, countCol


def getDestinationLocation(maxColorLocations, maxColorLocationsCnt, dataMatrixInt):

    r = maxColorLocations[0][0]
    c = maxColorLocations[0][1]
    code = dataMatrixInt[r][c]
    # print ">>>", maxColorLocations[0], "\tcolor _code:: ", code, "\t color :: ", const.ballColorCodeArray[code-1],
    # print "maxColorLocationsCnt :: ", maxColorLocationsCnt

    sumArray = np.empty(maxColorLocationsCnt, dtype=int)
    sumArrayCnt = 0
    if maxColorLocationsCnt == 1:
        l = maxColorLocations[0]
        top, bottom, left, right, sameCountRow, sameCountCol = getEmptyCellCount(
            l, dataMatrixInt)

        return False, l, l
    else:
        for i in range(0, maxColorLocationsCnt):
            l = maxColorLocations[i]
            top, bottom, left, right, sameCountRow, sameCountCol = getEmptyCellCount(
                l, dataMatrixInt)
            sum = top+bottom+right+left+(sameCountCol+sameCountRow)
            sumComplement = 4*const.GRID_SIZE-(top+bottom+right+left)
            sumArray[sumArrayCnt] = sum
            sumArrayCnt += 1

            # print "l :: ", l, "\t top :: ", top, "\t bottom :: ", bottom, "\t left :: ", left, "\t right :: ", right, "\t sameCountRow :: ", sameCountRow, "\t sameCountCol :: ", sameCountCol, "\t sum :: ", sum
            # print "sumComplement :: ", sumComplement

    maxIndex = sumArray.argmax()
    minIndex = sumArray.argmin()

    source = np.zeros(1, dtype=utils.cellDT)
    destination = np.zeros(1, dtype=utils.cellDT)

    source[0][0] = maxColorLocations[maxIndex][0]
    source[0][1] = maxColorLocations[maxIndex][1]

    destination[0][0] = maxColorLocations[minIndex][0]
    destination[0][1] = maxColorLocations[minIndex][1]

    top, bottom, left, right, sameCountRow, sameCountCol = getEmptyCellCount(
        [destination[0][0], destination[0][1]], dataMatrixInt)

    row = destination[0][0]
    col = destination[0][1]

    if top >= max(bottom, right, left):
        row -= 1
    elif bottom >= max(top, right, left):
        row += 1
    elif right >= max(top, bottom, left):
        col += 1
    elif left >= max(top, bottom, right):
        col -= 1

    finalDestination = np.zeros(1, dtype=utils.cellDT)
    finalDestination[0][0] = row
    finalDestination[0][1] = col

    # print "top :: ", top, "\t bottom :: ", bottom, "\t left :: ", left, "\t right :: ", right, "\t sameCountRow :: ", sameCountRow, "\t sameCountCol :: ", sameCountCol,
    # print "source :: ", source, "\t destination :: ", finalDestination

    return True, source, finalDestination


def makeRandomMove(dataMatrixInt, binaryMatrix):

    occupiedArray, emptyArray = getEmptyAndOccupiedCells(dataMatrixInt)

    totalOccupied = const.TOTAL_GRID_CELLS - \
        getNumberOfElementsInArray(occupiedArray)
    totalEmpty = const.TOTAL_GRID_CELLS + 1 - \
        getNumberOfElementsInArray(emptyArray)

    occupiedArray, emptyArray = getEmptyAndOccupiedCells(dataMatrixInt)

    totalOccupied = const.TOTAL_GRID_CELLS - \
        getNumberOfElementsInArray(occupiedArray)
    totalEmpty = const.TOTAL_GRID_CELLS + 1 - \
        getNumberOfElementsInArray(emptyArray)

    while 1:
        sRandom = random.randint(0, totalOccupied-1)
        dRandom = random.randint(0, totalEmpty-1)

        source = np.zeros(1, dtype=utils.cellDT)

        source[0][0] = occupiedArray[sRandom][0]
        source[0][1] = occupiedArray[sRandom][1]

        destination = np.zeros(1, dtype=utils.cellDT)
        destination[0][0] = emptyArray[dRandom][0]
        destination[0][1] = emptyArray[dRandom][1]

        flag = isPathExisting.has_path(
            binaryMatrix, source, destination)

        if flag:
            return source, destination


def getSourceLocation(destination, maxColorLocations, maxColorLocationsCnt, dataMatrixInt):
    source = np.zeros(1, dtype=utils.cellDT)

    return source


def isNonRandomMovePossible(dataMatrixInt, binaryMatrix, maxColorCode):

    maxColorLocations, maxColorLocationsCnt = getLocationsFor(
        maxColorCode, dataMatrixInt)

    # print "locations :: ", maxColorLocations

    flag, source, destination = getDestinationLocation(
        maxColorLocations, maxColorLocationsCnt, dataMatrixInt)

    if not flag:
        return flag, [(0, 0)], [(0, 0)]

    if isPathExisting.isPathExisting(source, destination, binaryMatrix):
        flag = True
    else:
        flag = False

    # print "flag :: ", flag, "\t source :: ", source, "\t destination :: ", destination
    return flag, source, destination


def versionOne(dataMatrixInt, binaryMatrix, gui):
    """ count the number of balls with most colour and process it... """

    colourCountArray, totalBalls = getColorCount(dataMatrixInt)
    # print "dataMatrix \n", dataMatrixInt
    # print "colourCountArray :: ", colourCountArray, "\t totalBalls :: ", totalBalls

    flag = False
    tryCnt = 0
    randomCnt = 0
    nonRandomCnt = 0

    arr1 = colourCountArray[:7]
    while tryCnt < 3:
        # print "arr1 :: ", arr1
        maxColorCode = arr1.argmax()
        color = const.ballColorCodeArray[maxColorCode]
        maxColorCode += 1

        # print "tryCnt :: ", tryCnt, "\t Selecting Ball :: ", color, " >> (", maxColorCode, ")"
        flag, source, destination = isNonRandomMovePossible(
            dataMatrixInt, binaryMatrix, maxColorCode)

        if flag:
            # print "source :: ", source, "\t desination :: ", destination
            break

        arr1[maxColorCode-1] = 0
        tryCnt += 1
        # a = raw_input()

    if not flag:
        # print "Making-random-move..."
        source, destination = makeRandomMove(
            dataMatrixInt, binaryMatrix)
        randomCnt += 1
    else:
        # print "Making non-Random Move..."
        nonRandomCnt += 1

    # a = raw_input()
    gui.moveCursorFromTo(source, destination)
    # print "randomCnt :: ", randomCnt, "\t nonRandomCnt :: ", nonRandomCnt


def isBallForCodePresent(code, dataMatrixInt):

    for i in range(0, const.GRID_ROW):
        for j in range(0, const.GRID_COL):
            if dataMatrixInt[i][j] == code:
                return True

    return False


def uniqueBallCount(dataMatrixInt):
    cnt = 0

    for c in range(0, 7):
        if isBallForCodePresent(c+1, dataMatrixInt):
            cnt += 1

    return cnt


def getTopThreeColorInfo(dataMatrixInt):

    one = utils.colorLocationInfo()
    two = utils.colorLocationInfo()
    three = utils.colorLocationInfo()

    colourCountArray, totalBalls = getColorCount(dataMatrixInt)
    uniqueBalls = uniqueBallCount(dataMatrixInt)

    if totalBalls > 0:
        arr = colourCountArray[:7]
        oneCode = arr.argmax()+1
        one.colorCode = oneCode
        arr[arr.argmax()] = 0

        one.locationInfo, one.locationInfoCnt = getLocationForCode(
            oneCode, dataMatrixInt)

        twoCode = -1
        if uniqueBalls > 1:
            twoCode = arr.argmax()+1
            two.colorCode = twoCode
            two.locationInfo, two.locationInfoCnt = getLocationForCode(
                twoCode, dataMatrixInt)
            arr[arr.argmax()] = 0

        threeCode = -1
        if uniqueBalls > 2:
            threeCode = arr.argmax()+1
            three.colorCode = threeCode
            three.locationInfo, three.locationInfoCnt = getLocationForCode(
                threeCode, dataMatrixInt)

    return one, two, three


def getTopThreeColorInfo_v2(matrix):
   # Flatten the matrix and filter out -1
    filtered_values = matrix.flatten()
    filtered_values = filtered_values[filtered_values != -1]

    # Count occurrences of each unique value
    unique_values, counts = np.unique(filtered_values, return_counts=True)

    # Sort values based on counts in descending order
    sorted_indices = np.argsort(-counts)  # Sort in descending order
    unique_values_sorted = unique_values[sorted_indices]

    # Return the top three most frequent values (excluding -1)
    return list(unique_values_sorted[:3])


def areAllBallsInSameRowOrCol(c):

    cnt = 0
    for i in range(0, c.locationInfoCnt):
        for j in range(i+1, c.locationInfoCnt):
            if (c.locationInfo[i][0] == c.locationInfo[j][0]):
                cnt += 1

    # print "after row :: ", cnt
    # print "after locationInfoCnt :: ", c.locationInfoCnt
    if cnt == c.locationInfoCnt:
        return True

    cnt = 0
    for i in range(0, c.locationInfoCnt):
        for j in range(i+1, c.locationInfoCnt):
            if (c.locationInfo[i][1] == c.locationInfo[j][1]):
                cnt += 1

    if cnt == c.locationInfoCnt:
        return True
    else:
        return False


def getAvailableCellFor(row, col, dataMatrixInt):
    t = b = r = l = False

    # Top
    if row > 0:
        if dataMatrixInt[(row - 1)][col] == 0:
            t = True

    # Bottom
    if row < (const.GRID_ROW-1):
        if dataMatrixInt[(row + 1)][col] == 0:
            b = True

    # Right
    if col > 0:
        if dataMatrixInt[row][(col - 1)] == 0:
            r = True

    # Left
    if col < (const.GRID_COL-1):
        if dataMatrixInt[row][(col + 1)] == 0:
            l = True

    # print ">>> t :: ",t," b :: ",b," r :: ",r," l :: ",l
    return t, b, r, l


def isNonRandomMoveV2(dataMatrixInt, binaryMatrix, c):

    if c.locationInfoCnt == 1:
        return False, 0, 0

    if areAllBallsInSameRowOrCol(c):
        # print "Same row/col detected..."
        return False, 0, 0
    # else:
        # print "not is same row or col..."

    # T-B-R-L
    for i in range(0, c.locationInfoCnt):
        c.tbrl[i][0], c.tbrl[i][1], c.tbrl[i][2], c.tbrl[i][3], c.tbrl[i][4], c.tbrl[i][5] = getEmptyCellCount(
            c.locationInfo[i], dataMatrixInt)
        c.tbrl[i][6] = ((10*c.tbrl[i][4]) + (10 * c.tbrl[i][5])) + ((c.tbrl[i][0] + c.tbrl[i][1] +
                                                                     c.tbrl[i][2] + c.tbrl[i][3] + c.tbrl[i][4]) / 4)

    sourceLocationIndex = c.tbrl[:c.locationInfoCnt, 6:].argmin()

    source = np.zeros(1, dtype=utils.cellDT)
    # source = c.locationInfo[sourceLocationIndex]

    source[0][0] = c.locationInfo[sourceLocationIndex][0]
    source[0][1] = c.locationInfo[sourceLocationIndex][1]

    destination = np.zeros(1, dtype=utils.cellDT)
    destinationLocationIndex = c.tbrl[:c.locationInfoCnt, 6:].argmax()

    maxRowIndex = c.tbrl[:c.locationInfoCnt, 4:5].argmax()
    maxRowVal = c.tbrl[maxRowIndex][4]

    maxColIndex = c.tbrl[:c.locationInfoCnt, 5:6].argmax()
    maxColVal = c.tbrl[maxColIndex][5]

    row = -1
    col = -1
    finalRow = -1
    finalCol = -1

    if maxRowVal > maxColVal:  # going row wise...

        row = c.locationInfo[maxRowIndex][0]
        col = c.locationInfo[maxRowIndex][1]
        t, b, r, l = getAvailableCellFor(row, col, dataMatrixInt)

        finalRow = row
        if r:
            finalCol = col+1
        elif l:
            finalCol = col-1
    elif maxColVal > maxRowVal:  # going col wise...

        row = c.locationInfo[maxColIndex][0]
        col = c.locationInfo[maxColIndex][1]

        t, b, r, l = getAvailableCellFor(row, col, dataMatrixInt)

        finalCol = col
        if t:
            finalRow = row-1
        elif b:
            finalRow = row+1
    else:

        if (c.locationInfoCnt > 1):
            destinationLocationIndex = c.tbrl[:c.locationInfoCnt, 6:].argmax()

            row = c.locationInfo[destinationLocationIndex][0]
            col = c.locationInfo[destinationLocationIndex][1]
            t, b, r, l = getAvailableCellFor(row, col, dataMatrixInt)

            finalRow = row
            finalCol = col

            if t:
                finalRow = row - 1
            elif b:
                finalRow = row + 1
            elif r:
                finalCol = col + 1
            elif l:
                finalCol = col - 1

            destination[0][0] = finalRow
            destination[0][1] = finalCol
        else:
            return False, 0, 0

    destination[0][0] = finalRow
    destination[0][1] = finalCol

    flag = isPathExisting.isPathExisting(source, destination, binaryMatrix)

    return flag, source, destination


def versionTwo(dataMatrixInt, binaryMatrix, gui):

    c1, c2, c3 = getTopThreeColorInfo(dataMatrixInt)

    # print "Selecting Ball-Color :: ", const.ballColorCodeArray[c1.colorCode-1]
    flag, source, destination = isNonRandomMoveV2(
        dataMatrixInt, binaryMatrix, c1)

    if flag:
        return flag, source, destination
    elif not flag:
        # print "Selecting Ball-Color :: ", const.ballColorCodeArray[c2.colorCode-1]
        flag, source, destination = isNonRandomMoveV2(
            dataMatrixInt, binaryMatrix, c2)

        if flag:
            return flag, source, destination
        else:
            # print "Selecting Ball-Color :: ", const.ballColorCodeArray[c3.colorCode-1]
            flag, source, destination = isNonRandomMoveV2(
                dataMatrixInt, binaryMatrix, c3)

            if flag:
                return flag, source, destination

    # a = raw_input()
    return False, source, destination

# all non-random moves to be implemented here. They will start from ver_three_*


def versionThree_one(dataMatrixInt, binaryMatrix, gui):
    # Get the most dominant colour balls - 3 most dominant

    print(dataMatrixInt)

    val = getTopThreeColorInfo_v2(dataMatrixInt)
    c1 = val[0]
    c2 = val[1]
    c3 = val[2]
    # c1, c2, c3 = getTopThreeColorInfo(dataMatrixInt)

    print(
        f"c1 :: {const.ballColorCodeArray[c1] }\t c2 :: {const.ballColorCodeArray[c2]}\t c3 :: {const.ballColorCodeArray[c3]}")

    # what next ???

    source, destination = makeRandomMove(
        dataMatrixInt, binaryMatrix)

    return True, source, destination


def main():
    gui = aGUI.autoGUI()

    gameCnt = 0
    while gameCnt < 3:

        randomMovesCnt = 0
        nonRandomMoveCnt = 0
        moveCnt = 1

        ballColourFreqArray = []
        while moveCnt <= 23:

            dataMatrixInt, binaryMatrix = getDataMatrix()

            # count the frequency of balls on the grid - for stastical analysis
            # ballColourFreqArray = ballDistData.getBallDistributionData(
            #     dataMatrixInt, ballColourFreqArray)

            start_time = time.time()
            flag, source, destination = versionThree_one(
                dataMatrixInt, binaryMatrix, gui)
            # source, destination = makeRandomMove(
            #     dataMatrixInt, binaryMatrix)
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)

            flag = 1
            if not flag:
                randomMovesCnt += 1
                source, destination = makeRandomMove(
                    dataMatrixInt, binaryMatrix)
            else:
                nonRandomMoveCnt += 1

            print("move-cnt :: " + str(moveCnt) + "\t src. :: " + str(source) + "\t dest. :: " +
                  str(destination) + "\t flag :: " + str(flag)+"\t time :: "+str(time_taken) + " Sec.")

            gui.moveCursorFromTo(source, destination)
            moveCnt += 1

        gameCnt += 1
        print("randomMoves :: ", randomMovesCnt,
              "\t nonRandomMovesCnt :: ", nonRandomMoveCnt)

        # Print ballColourFreqArray for each game
        # print(ballColourFreqArray)

        time.sleep(3)
        # aGUI.pyautogui.press('space')
        aGUI.pyautogui.hotkey('ctrl', 'n')


main()


# cv2.imshow('Captured Image', rowImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
