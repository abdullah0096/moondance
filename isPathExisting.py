from collections import deque
import numpy as np
import utils as utils
import constants as const


def getPossibleMoves(cell, binaryMatrix):
    moves = np.full(4, -1, dtype=utils.cellDT)

    # PATCH : To avoid cell has values other than 0-9
    if not (cell[0] in range(0, const.GRID_ROW)) or not (cell[1] in range(0, const.GRID_COL)):
        return moves

    # TOP-case
    if cell[0] != 0:
        if binaryMatrix[cell[0]-1][cell[1]] == 1:
            moves[0][0] = cell[0]-1
            moves[0][1] = cell[1]

    if cell[0] != const.GRID_SIZE-1:
        if binaryMatrix[cell[0]+1][cell[1]] == 1:
            moves[1][0] = cell[0]+1
            moves[1][1] = cell[1]

    if cell[1] != 0:
        if binaryMatrix[cell[0]][cell[1]-1] == 1:
            moves[2][0] = cell[0]
            moves[2][1] = cell[1]-1

    if cell[1] != const.GRID_SIZE-1:
        if binaryMatrix[cell[0]][cell[1]+1] == 1:
            moves[3][0] = cell[0]
            moves[3][1] = cell[1]+1

    return moves


def getNumberOfElementsIn(possibleMoves):
    cnt = 0
    for p in possibleMoves:
        if p[0] != -1:
            cnt += 1
        else:
            break

    return cnt


def mergeMoves(possibleMoves, moves):

    possibleMovesSize = getNumberOfElementsIn(possibleMoves)
    cnt = 0

    # print "=================Start===================="
    # print "moves :: ", moves
    # print "possible moves :: \n", possibleMoves

    for m1 in moves:
        if m1[0] == -1:
            continue

        flag = True
        for i in range(0, possibleMovesSize):
            m2 = possibleMoves[i]

            if m1[0] == m2[0] and m1[1] == m2[1]:
                flag = False
                break

        if flag:
            possibleMoves[possibleMovesSize+cnt][0] = m1[0]
            possibleMoves[possibleMovesSize+cnt][1] = m1[1]
            cnt += 1

    # print "possible moves :: \n", possibleMoves

    # print "=================END===================="
    return possibleMoves


def isDestinationReached(moves, destination):

    for i in range(0, 4):
        m1 = moves[i]
        if m1[0] == destination[0][0] and m1[1] == destination[0][1]:
            return True

    return False


def isPathExisting(source, destination, binaryMatrix):

    print("In isPathExist...")
    print("source :: "+str(source)+"\t dest. :: " +
          str(destination)+"\n binMat :: \n"+str(binaryMatrix))

    possibleMovesArraySize = 300
    possibleMoves = np.full(possibleMovesArraySize, -1, dtype=utils.cellDT)
    possibleMovesCnt = 0

    possibleMoves[possibleMovesCnt][0] = source[0][0]
    possibleMoves[possibleMovesCnt][1] = source[0][1]

    whileCnt = 0
    while whileCnt < possibleMovesArraySize-1:
        moves = getPossibleMoves(possibleMoves[possibleMovesCnt], binaryMatrix)

        if isDestinationReached(moves, destination):
            print("out isPathExist... TRUE")
            return True

        possibleMovesCnt += 1
        possibleMoves = mergeMoves(possibleMoves, moves)
        # a = raw_input()
        whileCnt += 1

    print("out isPathExist... FALSE")
    return False


# Define a helper function to check if a cell is valid to move to


def is_valid(matrix, visited, row, col):
    num_rows, num_cols = matrix.shape
    return 0 <= row < num_rows and 0 <= col < num_cols and matrix[row, col] == 0 and not visited[row, col]

# Define the main function to check path existence


def has_path(matrix, source, destination):
    source_row, source_col = source['row'][0], source['col'][0]
    dest_row, dest_col = destination['row'][0], destination['col'][0]

    # If source and destination are the same, path exists trivially
    if source_row == dest_row and source_col == dest_col:
        return True

    num_rows, num_cols = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    visited[source_row, source_col] = True

    # Define possible movements: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Recursive DFS function
    def dfs(row, col):
        # If we reach the destination, return True
        if row == dest_row and col == dest_col:
            return True

        # Try all valid directions
        for d in directions:
            neighbor_row = row + d[0]
            neighbor_col = col + d[1]

            if is_valid(matrix, visited, neighbor_row, neighbor_col):
                visited[neighbor_row, neighbor_col] = True
                if dfs(neighbor_row, neighbor_col):
                    return True

        return False

    # Start DFS from the source cell
    return dfs(source_row, source_col)
