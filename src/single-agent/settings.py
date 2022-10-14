"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Single Agent RL models

This script will be used to generate the static or dynamic environments that the 
RL models will be learning in. To start we will use simple, static environments. 

This script can be run from the command line using: $ settings.py
"""

# imports 
from tkinter import W
import numpy as np


class StaticEnv():

    def __init__(self, board: np.array, start_position: tuple):
        self.board = board
        self.agent_positon = start_position
        self.board_x = self.board.shape[1]
        self.board_y = self.board.shape[0]

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
            if self.board[px][py] is not np.nan:
                successors.append(position)

        # fix a bug in the indexing by flipping the values in the tuples
        successors = [(x[1], x[0]) for x in successors]

        # print for testing REMOVE LATER
        for pos in successors:
            self.board[pos[0]][pos[1]] = "N"
        print(self.board)

        return successors


    def print_board(self):
        self.board[self.agent_positon[0]][self.agent_positon[1]] = "A"
        print(self.board)
        # TODO later use a package for better printing!

    def show_board_img(self):
        # TODO update this - stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
        w = self.board_x * 20
        h = self.board_y * 20

        self.board[self.agent_positon[0]][self.agent_positon[1]] = 999
        data = self.board
        # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
        img = Image.fromarray(data, 'RGB')
        # img.save('my.png')
        img.show()


def main():
    # create a demo board for testing 
    env = StaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]], dtype="object"), start_position=(2,0))
    env.print_board()
    env.get_successors()
    env.show_board_img()


if __name__ == "__main__":
    main()
