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
import random
import time

# imports
import numpy as np
import seaborn as sns

class QLearning():

    def __init__(self, env: np.array):
        self.env = env

    def solve(self):
        pass
        # TODO 


class SARSA():

    def __init__(self, problem_name: str, action_space: list, epsilon: float = 0.5, plot_q_map: bool = True):
        self.problem_name = problem_name
        self.epsilon = epsilon
        self.action_space = action_space
        self.plot_q_map = plot_q_map

    def get_action(self, Q_map: np.array, S: tuple, action_space: list):
        """
        Helper function that returns an action
        Epsilon Greedy action selector 
        """
        # random number between 0 and 1
        r = random.random()
        if r < self.epsilon: 
            # Choose A from S using the policy from the Q_map (epsilon greedy) 
            prob_vector = Q_map[:, S[0], S[1]]  # <-- we index by [(all actions), agent_y, agent_x]
            action = np.argmax(prob_vector)
        else:
            action = random.choice(self.action_space)

        return action


    def sarsa(self):

        print(f'SARSA Initiated! Please be patient, training can take a while.')

        # TODO load the board from a boards.py file 
        board = np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]])
        original_board = board.copy()
        board_x = board.shape[1]
        board_y = board.shape[0]

        # define the env parameters: 
        start_position = (2,0)
        goal_positions = ((0, 3), (1, 3))
        alpha = 0.1
        gamma = 0.9
    
        # use board to create initial Q-map 
        Q_map = np.random.rand(board_x, board_y, len(self.action_space))

        # iterate through all the action spaces 
        for action in range(Q_map.shape[-1]): 
            # and iterate through all the terminal positions
            for goal_pos in goal_positions:
                x = goal_pos[1]
                y = goal_pos[0]
                Q_map[action, y, x] = 0

        # --- SARSA Algorithm --- 
        n_episodes: int = 100

        # TODO use TQDM here for better visibility into progress
        for i in range(n_episodes):

            if i%10==0:
                print(f'...{i}/{n_episodes} complete...')

            # create an environment 
            solution = Solution(problem_name=self.problem_name, model_name='SARSA')
            env = StaticEnv(board=board, start_position=start_position, solution=solution, 
                            goal_positions=goal_positions, action_space=self.action_space, 
                            VERBOSE=False)

            # Choose start position (this has already been chosen, see env() above)
            S = env.agent_positon
            env.initialize_agent()

            # get action
            A = self.get_action(Q_map=Q_map, S=S, action_space=self.action_space)

            while env.solution.solved is False: 
                # take the action
                env.make_move(action=A)
                # env.print_board()
                
                # define the reward and the new state
                R = original_board[env.agent_positon[0], env.agent_positon[1]]
                S_prime = env.agent_positon

                # choose the next action (from S_prime) based on Q_map
                A_prime = self.get_action(Q_map=Q_map, S=S_prime, action_space=self.action_space)

                # update the Q_map
                Q_map[A, S[0], S[1]] = Q_map[A, S[0], S[1]] + \
                                       alpha * (R + (gamma * Q_map[A_prime, S_prime[0], S_prime[1]])  - Q_map[A, S[0], S[1]])

                # update S and A
                S = S_prime
                A = A_prime 

        # print(f'Complete Q_map: {Q_map}')

        mean_q_map: np.array = np.mean(Q_map, axis=0)
        print(f'Stacked Q_map: \n {mean_q_map}')

        if self.plot_q_map: 
            sns.heatmap(mean_q_map, annot=True, linewidth=.5, cmap="crest")


        # TEST CODE - TODO this code block should be moved into a method that can be called to test random actions 
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
        print('something')


def main():

    clf = SARSA(problem_name='Static Goal Seek', action_space = [0, 1, 2, 3])
    clf.sarsa()

if __name__ == "__main__":
    main()
