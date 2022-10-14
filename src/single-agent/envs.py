"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Single Agent RL models

This script will be used to generate the static or dynamic environments that the 
RL models will be learning in. To start we will use simple, static environments. 

This script can be run from the command line using: $ envs.py
"""

# imports 
import numpy as np
import random
import time
import math
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime

# local imports 
from routines import write_animation


from settings import Configs, PATH_TO_MP4S


class StaticEnv():

    def __init__(self, board: np.array, start_position: tuple, VERBOSE: bool = True):
        self.board = board
        self.empty_board = board.copy()
        self.agent_positon = start_position
        self.board_x = self.board.shape[1]
        self.board_y = self.board.shape[0]
        self.VERBOSE = VERBOSE
        self.big_board = self.expand_board(board=self.board)

    def get_successors(self) -> tuple:
        """
        Return the neighbors of the agent's location given a 4 neighbor model. 

        --- NOTE --- 
        Numpy assumes that the (0, 0) index is in the upper left corner. Use the below matrix as a guide to understand
        how indexing works here: 

        [(0,0), (0,1), ..., (0,n)]
        [(1,0), (1,1), ..., (1,n)]
        [ ...    ...         ... ]
        [(m,0), (m,1), ..., (m,n)]

        :param: agent_location - 2-length tuple representing the zero indexed location of the agent on the grid
        :return: tuple representing the successors of the agent's current location
        """

        # get all neighbors using the 4 neighbor model
        x = self.agent_positon[1]
        y = self.agent_positon[0]

        # get all the neighbors in the 8 neighbor model 
        # ----- vvv This code snipet was borrowed from a project that was completed last year (it was modified slighly for this project) 
        s1: list = [(x2, y2) for x2 in range(x-1, x+2)
                                            for y2 in range(y-1, y+2)
                                            if (-1 < x < self.board_x and -1 < y < self.board_y and
                                                (x != x2 or y != y2) and (0 <= x2 < self.board_x) and (0 <= y2 < self.board_y))]
        # ----- ^^^

        # remove neighbors that arent in the 4 neighbor model (we could actualy remove this code later to use the 8 neighbor model)
        s2 = []
        for neighbor in s1:
            if (neighbor[0] == self.agent_positon[1]) | (neighbor[1] == self.agent_positon[0]):
                s2.append(neighbor)

        # remove blocking squares (the agent can't move to the squares represented by np.nan)
        successors = []
        for position in s2:
            px = position[1]
            py = position[0]
            if np.isnan(self.board[px][py]):
                pass  # don't append the nans! 
            else:
                successors.append(position)

        # fix a bug in the indexing by flipping the values in the tuples
        successors = [(x[1], x[0]) for x in successors]

        return successors

    @staticmethod
    def expand_board(board: np.array):
        """
        This function expands the np.array by several orders of magnitude so that it can be plotted nicely. 
        """
        X = board.shape[1] * Configs.BOARD_INCREASE
        Y = board.shape[0] * Configs.BOARD_INCREASE

        big_board: np.array = np.zeros([Y, X])

        for i in range(big_board.shape[1]):
            for j in range(big_board.shape[0]):
                # find index in self.board
                x = math.floor(i/Configs.BOARD_INCREASE)
                y = math.floor(j/Configs.BOARD_INCREASE)
                big_board[j][i] = board[y][x]

        return big_board


    def print_board(self):
        self.board[self.agent_positon[0]][self.agent_positon[1]] = 50
        print(self.board)

    def make_move(self, new_pos: tuple):
        old_pos = self.agent_positon
        self.agent_positon = new_pos
        self.board[old_pos[0]][old_pos[1]] = 0
        if self.VERBOSE: 
            self.print_board()

    def random_move(self):
        # get a random move from the get successors method
        new_pos = random.choice(self.get_successors())
        
        # collect the old position and value so we can replace them after the move 
        old_pos = self.agent_positon
        old_value = self.empty_board[old_pos[0]][old_pos[1]]
        
        # update agent position and update old value 
        self.agent_positon = new_pos
        self.board[old_pos[0]][old_pos[1]] = old_value


    def show_board_img(self):
        self.board[self.agent_positon[0]][self.agent_positon[1]] = 50
        plt.imshow(self.board, interpolation='nearest')
        plt.show()


def main():
    # create a demo board for testing 
    # env = StaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]], dtype="object"), start_position=(2,0))
    env = StaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]]), start_position=(2,0))
  
    # some test code 
    env_snapshots = []
    for i in range(25):
        env.print_board()
        if Configs.generate_video:
            big_board = env.expand_board(board=env.board)
            for j in range(5):
                env_snapshots.append(big_board)
        env.random_move()
        time.sleep(0.2)

    if Configs.generate_video:
        env_itr = iter(env_snapshots)

        # write the animation file 
        now = datetime.now()
        timestamp = now.strftime("%H-%M-%S")
        mov_fname: str = f'static-env-{timestamp}.mp4'
        MOV_PATH = PATH_TO_MP4S / mov_fname
        write_animation(itr=env_itr, out_file=MOV_PATH)


# some test code
if __name__ == "__main__":
    main()
