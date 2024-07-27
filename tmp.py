import numpy as np

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


# Example usage with structured numpy dtype:
matrix = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

npCell = np.dtype([('row', 'i1'), ('col', 'i1')])
cellDT = np.dtype(npCell)

# Define source and destination using structured dtype
# npCell = np.dtype([('row', 'i1'), ('col', 'i1')])
# source = np.array([(0, 6)], dtype=npCell)

destination = np.zeros(1, dtype=cellDT)
destination[0][0] = 1
destination[0][1] = 1

source = np.zeros(1, dtype=cellDT)
source[0][0] = 0
source[0][1] = 6


print(has_path(matrix, source, destination))  # Output: True
