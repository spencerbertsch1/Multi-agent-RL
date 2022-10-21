"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Single Agent RL models

This script contains implementations of single-agent reinforcement learning models such 
as Q-learning and SARSA. 

This script can be run from the command line using: $ models.py
"""
# local imports 
from envs import StaticEnv
from routines import Solution

# imports
import numpy as np
import time

class QLearning():

    def __init__(self, env: np.array):
        self.env = env

    def solve(self):
        pass
        # TODO 


class SARSA():

    def __init__(self, problem_name):
        self.problem_name = problem_name

    def sarsa(self):

        # TODO load the board from a boards.py file 
        board = np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]])
        
        # create a demo board for testing 
        solution = Solution(problem_name=self.problem_name, model_name='SARSA')
        env = StaticEnv(board=board, start_position=(2,0), solution=solution, goal_positions=((0, 3), (1, 3)))
    
        # use board to create initial Q-map 
        Q_map = np.random.rand(env.board_x, env.board_y, env.board[0].shape[0])

        # iterate through all the action spaces 
        for action in range(Q_map.shape[-1]): 
            # and iterate through all the terminal positions
            for goal_pos in env.goal_positions:
                x = goal_pos[1]
                y = goal_pos[0]
                Q_map[action, y, x] = 0

        # TODO implement the SARSA algorithm here

        # some test code 
        for i in range(50):
            print(f'CURRENT POS: {env.agent_positon}, SUCCESSORS: {env.get_successors()}')
            env.print_board()
            env.random_move()
            if env.solution.solved is True:
                print(env.solution)
                break
            time.sleep(0.1)


def main():

    clf = SARSA(problem_name='Static Goal Seek')
    clf.sarsa()

if __name__ == "__main__":
    main()
