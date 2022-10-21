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
        board_x = board.shape[1]
        board_y = board.shape[0]

        # define the env parameters: 
        start_position = (2,0)
        goal_positions = ((0, 3), (1, 3))
        action_space = [0, 1, 2, 3]  # north, east, south, west
        alpha = 0.1
    
        # use board to create initial Q-map 
        Q_map = np.random.rand(board_x, board_y, len(action_space))

        # iterate through all the action spaces 
        for action in range(Q_map.shape[-1]): 
            # and iterate through all the terminal positions
            for goal_pos in goal_positions:
                x = goal_pos[1]
                y = goal_pos[0]
                Q_map[action, y, x] = 0

        # --- SARSA Algorithm --- 
        n_episodes: int = 5

        for i in range(n_episodes):

            # create an environment 
            solution = Solution(problem_name=self.problem_name, model_name='SARSA')
            env = StaticEnv(board=board, start_position=start_position, solution=solution, 
                            goal_positions=goal_positions, action_space=action_space)

            # Choose start position (this has already been chosen, see env() above)
            S = env.agent_positon
            env.initialize_agent()

            # Choose A from S using the policy from the Q_map (epsilon greedy) 
            A = 0  # TODO add function to sample actions based on Q_map 

            while env.solution.solved is False: 
                # take the action
                env.make_move(action=A)
                env.print_board()
                # time.sleep(0.3)
                
                # TODO define the reward and the new state
                R = 0
                S_prime = 0

                # choose the next action (from S_prime) based on Q_map
                A_prime = 0  # TODO add function to sample actions based on Q_map 

                # update the Q_map
                # Q_map[S, A] = Q_map[S, A] + alpha * (R + (gamma * Q_map[S, A])  - Q_map[S, A])

                # update S and A
                S = S_prime
                A = A_prime 

        print(f'Complete Q_map: {Q_map}')


        """
        # RANDOM MOVE CODE (for testing) 
        # some test code (move randomly)
        # create an environment 
        solution = Solution(problem_name=self.problem_name, model_name='SARSA')
        env = StaticEnv(board=board, start_position=start_position, solution=solution, 
                        goal_positions=goal_positions, action_space=action_space)

        # Choose start position (this has already been chosen, see env() above)
        S = env.agent_positon
        env.initialize_agent()
        for i in range(50):
            print(f'CURRENT POS: {env.agent_positon}, SUCCESSORS: {env.get_successors()}')
            env.print_board()
            env.random_move()
            if env.solution.solved is True:
                print(env.solution)
                break
            time.sleep(0.3)
        """


def main():

    clf = SARSA(problem_name='Static Goal Seek')
    clf.sarsa()

if __name__ == "__main__":
    main()
