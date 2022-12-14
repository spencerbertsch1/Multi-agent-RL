"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Multi Agent RL models

This script will be used to store the static or dynamic "boards" that act as the basis for the 
environment object generated by envs.py. 
"""

# imports 
import numpy as np 

class board1:
    board = np.array([[0, 0, 0, 100], 
                      [0, np.nan, 0, -100], 
                      [0, 0, 0, 0]])
    original_board = board.copy()
    board_x = board.shape[1]
    board_y = board.shape[0]
    start_position = (2, 0, 2, 2)  # (2, 0, 2, 1)
    goal_positions = ((0, 3), (1, 3))


# class board2:
#     board = np.array([[0, 0, 0, 100], 
#                       [0, np.nan, 0, -100], 
#                       [0, 0, 0, -100],
#                       [np.nan, 0, 0, -100],
#                       [0, 0, 0, -100],])
#     original_board = board.copy()
#     board_x = board.shape[1]
#     board_y = board.shape[0]
#     start_position = (4,0)
#     goal_positions = ((0, 3), (1, 3), (2, 3), (3, 3), (4, 3))


# class board3:
#     board = np.array([[np.nan, 0, 0, 0, 100], 
#                       [0, 0, np.nan, np.nan, np.nan], 
#                       [0, 0, 0, 0, 0],
#                       [np.nan, np.nan, 0, np.nan, 0],
#                       [0, 0, 0, 0, 0],
#                       [0, np.nan, np.nan, 0, np.nan],
#                       [0, 0, 0, 0, -100],])
#     original_board = board.copy()
#     board_x = board.shape[1]
#     board_y = board.shape[0]
#     start_position = (6,0)
#     goal_positions = ((0, 4), (6, 4))
