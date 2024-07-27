import numpy as np
import constants as const

npCell = np.dtype([('row', 'i1'), ('col', 'i1')])
cellDT = np.dtype(npCell)


class Cell:
    row = 0
    col = 0

    def __init__(self, r, c):
        self.row = r
        self.col = c

    def printRowCol(self):
        print("row :: ", self.row, "\t col :: ", self.col)


class colorLocationInfo:
    colorCode = -1
    locationInfo = np.full(const.LOCATION_ARRAY_COUNT, -1, dtype=cellDT)
    locationInfoCnt = 0

    # T-B-R-L-sameRow-sameCol-avg(t+b+r+l)
    tbrl = np.full([const.TOTAL_GRID_CELLS, 7], -1, dtype=float)

    def __init__(self):
        pass
