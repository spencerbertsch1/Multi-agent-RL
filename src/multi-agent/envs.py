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
import itertools

# local imports 
from routines import write_animation, Solution

from mdp.settings import Configs, PATH_TO_MP4S


class MultiAgentStaticEnv():

    move_mapper: dict = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
    text_move_mapper: dict = {0: 'north', 1: 'east', 2: 'south', 3: 'west'}

    def __init__(self, board: np.array, start_position: tuple, goal_positions: tuple, 
                 solution: Solution, action_space: list, VERBOSE: bool = True):
        self.board = board
        self.empty_board = board.copy()
        self.solution = solution
        self.agent_positon = start_position
        self.num_agents = self.get_num_agents(start_position=start_position)
        self.goal_positions = goal_positions
        self.action_space = action_space
        self.board_x = self.board.shape[1]
        self.board_y = self.board.shape[0]
        self.VERBOSE = VERBOSE
        self.big_board = self.expand_board(board=self.board)
        
    def get_num_agents(self, start_position: tuple):
        """
        Simple function that returns the number of agents on the board
        """
        assert len(start_position)%2==0, f'The start position needs to describe the (i,j) coordinate of each agent and \
                                     therefor must be an even number. len({start_position}) is not even. Please update.'

        # TODO we should add another assert here that checks to make sure the agents are not initialized on top of goal nodes 
        # or blocker nodes. For now we can just initialize the agents with care. 

        return len(start_position)%2

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

        --- MULTI-AGENT IMPLEMENTATION ---

        Here we now want to return a tuple of length self.num_agents * 2. This tuple will be of the form:
        (agent1_i, agent1_j, agent2_i, agent2_j, ... , agent_phi_i, agent_phi_j) where there are phi agents. 
        """

        # iterates over all agents
        single_agent_successors = []
        for i in range(0, len(self.agent_positon), 2):
            # get all neighbors using the 4 neighbor model
            x = self.agent_positon[i+1]
            y = self.agent_positon[i]

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
                if (neighbor[0] == x) | (neighbor[1] == y): 
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

            single_agent_successors.append(successors) 

        # Create the (num_agents*2)-length tuples of successors 
        # NOTE: This problem now becomes combinatorial. A large number of agents will no doubt slow this function down
        successor_combos = list(itertools.product(*single_agent_successors))

        # remove all states in which agents sit on top of one another 
        legal_combos = []
        for combo in successor_combos:
            if len(set(combo)) < len(combo):
                pass # <-- here at least two robots overlap - this is not allowed
            else:
                legal_combos.append(combo)

        # lastly we need to combine the tuples into (num_agents*2)-length tuple states
        # FIXME this part of the function should be sped up - if we could vectorize this that would be great
        final_successors = []
        for action_combo in legal_combos:
            new_state = []
            for single_agent_pos in action_combo:
                for pos in single_agent_pos:
                    new_state.append(pos)
            state_tuple: tuple = tuple(new_state)
            final_successors.append(state_tuple)

        return final_successors

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

    def initialize_agents(self):
        """
        simple function to initialize the agent's position on the board 

        Remember here the start position will be an even-length tuple in which each 2-tuple describes the starting 
        coordinate of one agent on the board. 
        """
        for i in range(0, len(self.agent_positon), 2):
            y = self.agent_positon[i]
            x = self.agent_positon[i+1]
            self.board[y, x] = 999

    def print_board(self):
        print(self.board, '\n')

    def make_move(self, new_position: tuple):

        # make the move 
        old_pos = self.agent_positon
        self.agent_positon = new_position

        # set the old positions to zero 
        for i in range(0, len(old_pos), 2):
            y = old_pos[i]
            x = old_pos[i+1]
            self.board[y, x] = 0

        # set the new positions to 99x... 
        for i in range(0, len(new_position), 2):
            y = new_position[i]
            x = new_position[i+1]
            # we subtract 1 here so that we can tell the agents apart! 
            self.board[y, x] = 999-(i/2)

        if self.VERBOSE: 
            self.print_board()

        self.solution.path += self.agent_positon
        self.solution.nodes_visited += 1
        self.solution.steps += 1

        # TODO remove agents individually if they get to the goal ! for i in range(0, len(old_pos), 2):
        if self.agent_positon in self.goal_positions:
            self.solution.solved = True
        #     self.solution.reward = self.empty_board[self.agent_positon[0]][self.agent_positon[1]]
        #     print(f'Game is over, final reward: {self.solution.reward}.')


        
    def random_move(self, successors: list):

        # get a random action
        new_position = random.choice(successors)

        self.make_move(new_position=new_position)
        

def make_movie():
    """
    Simple funciton that generates and saves an mp4 file showing the agent's progrssion through the environment
    """
    # create a demo board for testing 
    # env = StaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]], dtype="object"), start_position=(2,0))
    env = MultiAgentStaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]]), start_position=(2,0))
  
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


def random_walk():
    # create a demo board for testing 
    # env = StaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]], dtype="object"), start_position=(2,0))
    env = MultiAgentStaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]]), start_position=(2,0), goal_positions=((0, 3), (1, 3)))
  
    # some test code 
    env_snapshots = []
    for i in range(25):
        env.print_board()
        env.random_move()
        time.sleep(0.2)

def main():
    random_walk()

# some test code
if __name__ == "__main__":
    main()
