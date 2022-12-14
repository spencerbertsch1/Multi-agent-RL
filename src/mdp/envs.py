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
from routines import write_animation, Solution

from settings import MDP, PATH_TO_MP4S


class MDPStaticEnv():

    move_mapper: dict = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
    text_move_mapper: dict = {0: 'north', 1: 'east', 2: 'south', 3: 'west', 4: 'drop_phos_chek'}

    def __init__(self, board: np.array, agent_start_position: tuple, fire_start_position: tuple, 
                 solution: Solution, action_space: list, VERBOSE: bool = True, timestamp: int = 0):
        self.board = board
        self.agent_board = board.copy()
        self.empty_board = board.copy()
        self.solution = solution
        self.agent_position = agent_start_position
        self.fire_position = fire_start_position
        self.action_space = action_space
        self.board_x = self.board.shape[1]
        self.board_y = self.board.shape[0]
        self.VERBOSE = VERBOSE
        self.big_board = self.expand_board(board=self.board)
        self.timestamp = timestamp  # measured in base time units defined in the MDP formulation 
        

    def get_neighbors(self, node_location: tuple) -> tuple:
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
        x = node_location[1]
        y = node_location[0]

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
            if (neighbor[0] == self.agent_position[1]) | (neighbor[1] == self.agent_position[0]):
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
        X = board.shape[1] * MDP.board_increase
        Y = board.shape[0] * MDP.board_increase

        big_board: np.array = np.zeros([Y, X])

        for i in range(big_board.shape[1]):
            for j in range(big_board.shape[0]):
                # find index in self.board
                x = math.floor(i/MDP.board_increase)
                y = math.floor(j/MDP.board_increase)
                big_board[j][i] = board[y][x]

        return big_board

    def initialize_state(self):
        """
        simple function to initialize the agent's position and the fire on the board 
        """
        self.board = self.empty_board.copy()
        # initialize the single agent - later this can be done in a loop for more agents
        self.agent_board[self.agent_position[0], self.agent_position[1]] = 999
        # initialize the fire - later this can be done in a loop for more fire start locations
        self.board[self.fire_position[0], self.fire_position[1]] = 1

    def fire_spread(self):
        """
        Utility function that is used to spread the fire via either a stochastic or deterministic model 
        """
        if MDP.stochastic_fire_spread: 
            pass  # implement the stochastic fire simulation later
        else:

            # get burning node location
            old_node_location = np.where(self.board == 1)
            old_node_location_tuple = (old_node_location[0][0], old_node_location[1][0])

            # new burning node location
            new_burning_node_location = tuple([i+1 for i in old_node_location_tuple])

            # now we need to test whether or not the fire has burned out 
            if (new_burning_node_location[0] >= self.board_y) | (new_burning_node_location[1] >= self.board_x): 
                self.solution.solved = True
                if self.VERBOSE:
                    print('The fire has gone out! This episode is now complete.')

            elif (self.board[new_burning_node_location[0], new_burning_node_location[1]] == 9): 
                self.solution.solved = True
                if self.VERBOSE:
                    print('The fire has gone out! This episode is now complete.')
            else:
                # update the board to reflect the burn 
                self.board[new_burning_node_location[0], new_burning_node_location[1]] = 1
                self.board[old_node_location[0], old_node_location[1]] = 2

    def print_board(self):
        """
        Simple utility function to pretty print the board 
        """
        board_to_print: np.array = self.board.copy() + self.agent_board.copy()
        s = '-'*30
        print(f'{s}\n {board_to_print}\n {s}')

    def take_action(self, action: int):
        """
        This method updates the state of the system given an action that was taken at time (t). This method
        return the updated state at time (t+1). 
        """

        assert action in self.action_space, f'Action needs to be in {self.action_space}, but {action} was passed instead.'

        # get the new position from the action_mapper dict 
        if self.VERBOSE:
            print(f'ACTION: {self.text_move_mapper[action]}')

        if action == 4: 
            # we can't drop phos chek on already burned or burning nodes
            if (self.board[self.agent_position[0], self.agent_position[1]] == 2) | \
               (self.board[self.agent_position[0], self.agent_position[1]] == 1):
                pass
            else:
                # here we modify the phos_chek board so that the agent's current position is a 9 
                self.board[self.agent_position[0], self.agent_position[1]] = 9

        else:
            # if the action is not legal, we stay in the same position and burn no fuel 
            new_position = tuple(sum(x) for x in zip(self.agent_position, self.move_mapper[action]))

            # get a random move from the get successors method
            if new_position in self.get_neighbors(node_location=self.agent_position):
                # make the move 
                old_pos = self.agent_position
                self.agent_position = new_position
                self.agent_board[old_pos[0]][old_pos[1]] = 0
                self.agent_board[new_position[0]][new_position[1]] = 999
                if self.VERBOSE: 
                    self.print_board()

                self.solution.path += self.agent_position
                self.solution.nodes_visited += 1
                self.solution.steps += 1

            else: 
                # agent stays in the same place and burns no fuel 
                self.solution.steps += 1

    def random_move(self):

        # get a random action
        action = random.choice(self.action_space)

        self.take_action(action=action)
        

    def increment_time(self, action: int):
        """
        This method increments the environment forward by one base time unit. 
        If the agent moves every 2 time units and the fire moves every 10 time units, this is the method
        that keeps track of this information, logging the time and moving the state of the system forward. 
        """

        # increment base time unit by 1. 
        self.timestamp += 1
        
        if self.timestamp % MDP.aircraft_update_window == 0: 
            # here we initiate an action for the agent
            self.take_action(action=action)

        if self.timestamp % MDP.wildfire_update_window == 0: 
            # here we initiate the stoachastic or deterministic action for the fire
            self.fire_spread()

    def calculate_final_reward(self):
        """
        Small method that returns the reward given the current state
        """
        # this reward funciton is based on the simplest MDP model, leter iterations can be more sophistocated
        return np.count_nonzero(self.board==0) + np.count_nonzero(self.board==9)

    def calculate_reward(self, action: int):
        """
        Small method that returns the reward given the current state, action pair
        """
        # reward for dropping phos chek in fire's path
        if ((action == 4) & (self.agent_position[0] == self.agent_position[1]) & (self.agent_position[0] <= 1)):
            # ^^ this second check is only useful because of the linear fire spread - remove for radial fire spread
            # here we calculate the reward based on the linear distance from the fire start location
            R = 10
            # here we center the linear surface on the position (1,1), so the maximal reward is achieved by dropping 
            # phos chek directly in front of the fire. Rewards for dropping phos check spread equally from this point. 
            abs_diff = abs(sum(self.agent_position) - 2)
            reward = R - abs_diff
            # print(reward)
            return reward
        else:
            return 0

def make_movie():
    """
    Simple funciton that generates and saves an mp4 file showing the agent's progrssion through the environment
    """
    # create a demo board for testing 
    # env = StaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]], dtype="object"), start_position=(2,0))
    env = MDPStaticEnv(board=np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]]), start_position=(2,0))
  
    # some test code 
    env_snapshots = []
    for i in range(25):
        env.print_board()
        if MDP.generate_mp4:
            big_board = env.expand_board(board=env.board)
            for j in range(5):
                env_snapshots.append(big_board)
        env.random_move()
        time.sleep(0.2)

    if MDP.generate_mp4:
        env_itr = iter(env_snapshots)

        # write the animation file 
        now = datetime.now()
        timestamp = now.strftime("%H-%M-%S")
        mov_fname: str = f'static-env-{timestamp}.mp4'
        MOV_PATH = PATH_TO_MP4S / mov_fname
        write_animation(itr=env_itr, out_file=MOV_PATH)


def random_walk():

    problem_name = 'MDP BUG CONTAINMENT'
 
    board = board=np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    action_space = [0, 1, 2, 3, 4]

    agent_start_position = (3,3)
    fire_start_position = (0,0)

    # create an environment 
    solution = Solution(problem_name=problem_name, model_name='SARSA')
    env = MDPStaticEnv(board=board, agent_start_position=agent_start_position, solution=solution, 
                    fire_start_position=fire_start_position, action_space=action_space, 
                    VERBOSE=False)

    # initialize the agent(s) and the fire start locations
    env.initialize_state()

    # Choose start position (this has already been chosen, see env() above)
    for i in range(50):
        print(f'CURRENT POS: {env.agent_position}, SUCCESSORS: {env.get_neighbors(node_location=env.agent_position)}')
        env.print_board()
        env.increment_time()
        # env.random_move()
        if env.solution.solved is True:
            env.solution.reward = env.calculate_final_reward()
            print(env.solution)
            break
        time.sleep(0.2)

def main():
    random_walk()

# some test code
if __name__ == "__main__":
    main()