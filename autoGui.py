import time
import pyautogui
from utils import Cell

import constants as const
pyautogui.FAILSAFE = True


# MANULE Caibiration
def calibirateXandYAxis():

    # Calibirating Horizontally
    x = 27
    y = 94
    pyautogui.moveTo(x, y, duration=2)
    pyautogui.click(x, y, clicks=1,
                    interval=1, button='left')

    whileCnt = 0
    while whileCnt < 9:
        x = x + 33
        y = 94
        pyautogui.moveTo(x, y, duration=0)
        whileCnt += 1
        time.sleep(1)

    # Calibirating Vertically
    x = 27
    y = 94
    pyautogui.moveTo(x, y, duration=2)

    whileCnt = 0
    while whileCnt < 9:
        x = 27
        y = y + 33
        pyautogui.moveTo(x, y, duration=0)
        whileCnt += 1
        time.sleep(1)


def cursorMoveDemo():
    x = const.TOPLEFT_X
    y = const.TOPLEFT_Y

    pyautogui.moveTo(x, y, duration=2)

    x += const.CURSOR_STEP_SIZE/2
    y += const.CURSOR_STEP_SIZE/2

    pyautogui.click(x, y, clicks=1,
                    interval=1, button='left')

    moveToRow = 5
    moveToCol = 8

    row = x + (const.CURSOR_STEP_SIZE * (moveToCol-1))
    col = y + (const.CURSOR_STEP_SIZE * (moveToRow-1))

    pyautogui.moveTo(row, col, const.CURSOR_MOVE_DURATION)
    pyautogui.click(row, col, clicks=1,
                    interval=1, button='left')

    pyautogui.click(27, 94, clicks=1,
                    interval=1, button='left')


class autoGUI:

    isGUI_Initialised = False
    x = 0
    y = 0

    def __init__(self):
        self.initAutoGUI()

    def initAutoGUI(self):
        self.x = const.TOPLEFT_X
        self.y = const.TOPLEFT_Y

        pyautogui.moveTo(self.x, self.y, const.CURSOR_MOVE_DURATION)

        self.x += const.CURSOR_STEP_SIZE/2
        self.y += const.CURSOR_STEP_SIZE/2

        pyautogui.moveTo(self.x, self.y, const.CURSOR_MOVE_DURATION)
        self.isGUI_Initialised = True
        print("Auto-GUI Initialized...")

    def moveCursorToCell(self, destination):
        row1 = self.x + (const.CURSOR_STEP_SIZE * (destination[0][1]))
        col1 = self.y + (const.CURSOR_STEP_SIZE * (destination[0][0]))
        pyautogui.moveTo(row1, col1, const.CURSOR_MOVE_DURATION)

    def moveCursorFromTo(self, source, destination):

        row1 = self.x + (const.CURSOR_STEP_SIZE * (source[0][1]))
        col1 = self.y + (const.CURSOR_STEP_SIZE * (source[0][0]))

        pyautogui.moveTo(row1, col1, const.CURSOR_MOVE_DURATION)
        pyautogui.click(row1, col1, clicks=1, interval=1, button='left')

        row2 = self.x + (const.CURSOR_STEP_SIZE * (destination[0][1]))
        col2 = self.y + (const.CURSOR_STEP_SIZE * (destination[0][0]))

        pyautogui.moveTo(row2, col2, const.CURSOR_MOVE_DURATION)
        pyautogui.click(row2, col2, clicks=1, interval=1, button='left')
