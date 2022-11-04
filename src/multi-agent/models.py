"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Multi Agent RL models

This script contains implementations of multi-agent reinforcement learning models. Here we apply the same models 
(Q-learning and SARSA) to the problem, but now there are multiple agents that move during every decision epoch (k). 

This script can be run from the command line using: $ models.py
"""
# local imports 
from envs import MultiAgentStaticEnv
from routines import Solution
from boards import board1
import random
import time

# imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class TDLearning():

    def __init__(self, board_obj, problem_name: str, action_space: list, epsilon: float, plot_q_map: bool,
                 alpha: float, gamma: float, n_episodes: int):
        self.board_obj = board_obj
        self.problem_name = problem_name
        self.epsilon = epsilon
        self.action_space = action_space
        self.plot_q_map = plot_q_map
        self.alpha = alpha
        self.gamma = gamma
        self.n_episodes = n_episodes

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

    
    def q_learning(self):
        """
        Implementation of Q-learning algorithm using custom environment 
        """
        print(f'Q-Learning Initiated! Please be patient, training can take a while.')

        pass


    def sarsa(self):

        print(f'SARSA Initiated! Please be patient, training can take a while.')

        # load the board from a boards.py file 
        board = self.board_obj.board
        original_board = self.board_obj.original_board
        board_x = self.board_obj.board_x
        board_y = self.board_obj.board_y
        start_position = self.board_obj.start_position
        goal_positions = self.board_obj.goal_positions

        # use board to create initial Q-map 
        # Q_map: [NUM_ACTIONS, BOARD_Y, BOARD_X]
        Q_map = np.random.rand(len(self.action_space), board_y, board_x)
        
        # initialize all the goal positions to zero 
        # iterate through all the action spaces 
        for action in range(Q_map.shape[0]): 
            # and iterate through all the terminal positions
            for goal_pos in goal_positions:
                x = goal_pos[1]
                y = goal_pos[0]
                Q_map[action, y, x] = 0

        # --- SARSA Algorithm --- 
        for i in range(self.n_episodes):

            if i%10==0:
                print(f'...SARSA {i}/{self.n_episodes} complete...')

            # create an environment 
            solution = Solution(problem_name=self.problem_name, model_name='SARSA')
            env = MultiAgentStaticEnv(board=board, start_position=start_position, solution=solution, 
                            goal_positions=goal_positions, action_space=self.action_space, 
                            VERBOSE=False)

            # Initialize the agent in the start position 
            S = env.agent_positon
            env.initialize_agent()

            # get action
            A = self.get_action(Q_map=Q_map, S=S, action_space=self.action_space)

            while env.solution.solved is False: 
                # take the action
                env.make_move(action=A)
                
                # define the reward and the new state
                R = original_board[env.agent_positon[0], env.agent_positon[1]]
                S_prime = env.agent_positon

                # choose the next action (from S_prime) based on Q_map
                A_prime = self.get_action(Q_map=Q_map, S=S_prime, action_space=self.action_space)

                # update the Q_map
                Q_map[A, S[0], S[1]] = Q_map[A, S[0], S[1]] + \
                                       self.alpha * (R + (self.gamma * Q_map[A_prime, S_prime[0], S_prime[1]])  - Q_map[A, S[0], S[1]])

                # update S and A
                S = S_prime
                A = A_prime 

        mean_q_map: np.array = np.mean(Q_map, axis=0)
        print(f'Stacked Q_map: \n {mean_q_map}')

        if self.plot_q_map: 
            sns.heatmap(mean_q_map, annot=True, linewidth=.5, cmap="crest")
            # TODO add title with num training episodes 
            plt.show()


def random_action_test(problem_name:str='multi_agent_static_goal_seek'):
    """
    Simple function that shows the current environment with an agent taking random actions
    """
    board = np.array([[0, 0, 0, 100], [0, np.nan, 0, -100], [0, 0, 0, 0]])
    action_space = [0, 1, 2, 3]

    start_position = (2,0,2,1)
    goal_positions = ((0, 3), (1, 3))

    # create an environment 
    solution = Solution(problem_name=problem_name, model_name='SARSA')
    env = MultiAgentStaticEnv(board=board, start_position=start_position, solution=solution, 
                    goal_positions=goal_positions, action_space=action_space, 
                    VERBOSE=False)

    # Choose start position (this has already been chosen, see env() above)
    S = env.agent_positon
    env.initialize_agents()
    for i in range(50):
        print(f'CURRENT POS: {env.agent_positon}, SUCCESSORS: {env.get_successors()}')
        env.print_board()
        env.random_move()
        if env.solution.solved is True:
            print(env.solution)
            break
        time.sleep(0.3)


def main():

    # define the SARSA model with all of the necessary environment parameters 
    clf = TDLearning(problem_name='Multi Agent Static Goal Seek', 
                board_obj = board1, 
                action_space = [0, 1, 2, 3], 
                plot_q_map=True,
                alpha = 0.1,
                gamma = 0.9, 
                epsilon = 0.5, 
                n_episodes = 50)
    
    # test SARSA
    clf.sarsa()

    # test Q-learning 
    # clf.q_learning()

if __name__ == "__main__":
    # main()
    random_action_test()
