import numpy as np

# create a module constant.py and import this module here

NUMBER_OF_BALLS = 9

# redBall = np.array([152,  99, 111])
redBall = np.array([106,  92, 145])

# orangeBall = np.array([145, 124, 103])
orangeBall = np.array([103, 123, 144])

# blueBall = np.array([85, 97, 176])
blueBall = np.array([171,  88,  76])

# yellowBall = np.array([152, 163, 113])
yellowBall = np.array([106, 158, 145])

# purpleBall = np.array([152, 99, 176])
purpleBall = np.array([172,  91, 145])

# greenBall = np.array([89, 163, 113])
greenBall = np.array([107, 158,  80])

# cyanBall = np.array([89, 163, 176])
cyanBall = np.array([171, 157,  80])


# noneBall = np.array([122, 146, 173])
noneBall = np.array([172, 142, 115])

ballColorCodeArray = np.array(
    ["RED", "BLUE", "ORANGE", "YELLOW", "PURPLE", "GREEN", "CYAN", "NO-BALL"])
#       0     1         2       3           4       5       6       7

ballColorShortCodeArray = np.array(
    ['R', 'B', 'O', 'Y', 'P', 'G', 'C', 'N'])


# top left of the grid
# TOP_LEFT_X = 20
# TOP_LEFT_Y = 105


# Change this to change the dimension of the board
CURSOR_STEP_SIZE = 101  # 9x9

# CURSOR_STEP_SIZE = 130   7x7

# bottom right end of the grid - these need not be changed in case board size changes
# these will work for 7x7 and 9x9

# BOTTOM_RIGHT_X = 484  # projector
# BOTTOM_RIGHT_Y = 637  # projector
# TOPLEFT_X = 24  # projector
# TOPLEFT_Y = 177 # projector
# CURSOR_STEP_SIZE = 50  # 9x9 projector

BOTTOM_RIGHT_X = 936
BOTTOM_RIGHT_Y = 1023
TOPLEFT_X = 20
TOPLEFT_Y = 105

CURSOR_MOVE_DURATION = 1
CLICK_DURATION = 0

COLOR_SIMILAR_FACTOR = 10

GRID_ROW = 9
GRID_COL = 9

GRID_SIZE = 9
TOTAL_GRID_CELLS = GRID_ROW*GRID_COL

LOCATION_ARRAY_COUNT = 3*TOTAL_GRID_CELLS
