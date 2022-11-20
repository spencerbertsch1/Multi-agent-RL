"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------
MDP Solution - Model Implementation
This script contains implementations of single-agent reinforcement learning models such 
as Q-learning and SARSA. 
This script can be run from the command line using: $ models.py
"""
# local imports 
from envs import MDPStaticEnv
from routines import Solution, write_animation
from boards import board1, board2
from settings import MDP, PATH_TO_MP4S, PATH_TO_POLICIES
import random
import time
import datetime

# imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use("seaborn-v0_8-notebook")

class PolicyIteration:

    def __init__(self):
        pass


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
        # position = S[0]
        # board_state = S[1]  # <-- we might want to use this to get the action later on

        # random number between 0 and 1
        r = random.random()
        if r < self.epsilon: 
            # Choose A from S using the policy from the Q_map (epsilon greedy) 
            prob_vector = Q_map[:, S[0], S[1]]  # <-- we index by [(all actions), agent_y, agent_x]
            action = np.argmax(prob_vector)
        else:
            action = random.choice(self.action_space)

        return action

    def get_max_axtion(self, Q_map: np.array, S: tuple, action_space: list):
        
        prob_vector = Q_map[:, S[0], S[1]]  # <-- we index by [(all actions), agent_y, agent_x]
        action = np.argmax(prob_vector)
        
        return action
        
        
    def q_learning(self):
        
        print(f'Q=Learning Initiated! Please be patient, training can take a while.')

        # load the board from a boards.py file 
        board = self.board_obj.board
        original_board = self.board_obj.original_board
        board_x = self.board_obj.board_x
        board_y = self.board_obj.board_y
        start_position = self.board_obj.start_position
        goal_positions = self.board_obj.goal_positions

        # use board to create initial Q-map 
        Q_map = np.random.rand(board_x, board_y, len(self.action_space))
        
        # initialize all the goal positions to zero 
        # iterate through all the action spaces 
        for action in range(Q_map.shape[-1]): 
            # and iterate through all the terminal positions
            for goal_pos in goal_positions:
                x = goal_pos[1]
                y = goal_pos[0]
                Q_map[action, y, x] = 0
                
        # --- Q-Learning Algorithm --- 
        for i in range(self.n_episodes):

            if i%10==0:
                print(f'...QLearning {i}/{self.n_episodes} complete...')

            # create an environment 
            solution = Solution(problem_name=self.problem_name, model_name='Q-Learn')
            env = MDPStaticEnv(board=board, start_position=start_position, solution=solution, 
                            goal_positions=goal_positions, action_space=self.action_space, 
                            VERBOSE=False)

            # Initialize the agent in the start position 
            S = env.agent_position
            env.initialize_agent()

            # get action
            A = self.get_action(Q_map=Q_map, S=S, action_space=self.action_space)

            while env.solution.solved is False: 
                # take the action
                env.make_move(action=A)
                
                # define the reward and the new state
                R = original_board[env.agent_position[0], env.agent_position[1]]
                S_prime = env.agent_position

                # choose the next action (from S_prime) based on Q_map
                A_prime = self.get_action(Q_map=Q_map, S=S_prime, action_space=self.action_space)
                
                A_max = self.get_max_action(Q_map=Q_map, S=S_prime, action_space=self.action_space)

                # update the Q_map
                Q_map[A, S[0], S[1]] = Q_map[A, S[0], S[1]] + \
                                       self.alpha * (R + (self.gamma * Q_map[A_max, S_prime[0], S_prime[1]])  - Q_map[A, S[0], S[1]])

                # update S and A
                S = S_prime
                A = A_prime 

        mean_q_map: np.array = np.mean(Q_map, axis=0)
        print(f'Stacked Q_map: \n {mean_q_map}')

        if self.plot_q_map: 
            sns.heatmap(mean_q_map, annot=True, linewidth=.5, cmap="crest")
            plt.show()
        

        """
        Implementation of Q-learning algorithm using custom environment 
        """
        print(f'Q-Learning Initiated! Please be patient, training can take a while.')



    def sarsa(self):

        print(f'SARSA Initiated! Please be patient, training can take a while.')

        # load the board from a boards.py file 
        board = self.board_obj.board
        original_board = self.board_obj.original_board
        board_x = self.board_obj.board_x
        board_y = self.board_obj.board_y

        # use board to create initial Q-map 
        # Q_map: [NUM_ACTIONS, BOARD_Y, BOARD_X]
        Q_map = np.random.rand(len(self.action_space), board_y, board_x)
         
        # --- SARSA Algorithm --- 
        reward_tracker = []
        for i in range(self.n_episodes):

            if i%10==0:
                print(f'...SARSA {i}/{self.n_episodes} complete...')

            problem_name = 'MDP BUG CONTAINMENT'

            agent_start_position = (3,3)
            fire_start_position = (0,0)

            # create an environment 
            solution = Solution(problem_name=problem_name, model_name='SARSA')
            env = MDPStaticEnv(board=board, agent_start_position=agent_start_position, solution=solution, 
                            fire_start_position=fire_start_position, action_space=self.action_space, 
                            VERBOSE=MDP.verbose)

            # initialize the agent(s) and the fire start locations
            env.initialize_state()
            S = env.agent_position
            # S = (env.agent_position, env.board)  # <- we may need the board in the state later 

            # get action
            A = self.get_action(Q_map=Q_map, S=S, action_space=self.action_space)

            idx = 0
            while env.solution.solved is False: 
                idx += 1
                # take the action
                env.increment_time(action=A)
                
                # define the reward and the new state
                R = env.calculate_reward(action=A)  # <-- we need to update this so it will work for SARSA
                S_prime = env.agent_position

                if env.VERBOSE:
                    print(f'BOARD: \n {env.board} \n ACTION: {env.text_move_mapper[A]}, REWARD: {R}')

                # choose the next action (from S_prime) based on Q_map
                A_prime = self.get_action(Q_map=Q_map, S=S_prime, action_space=self.action_space)

                # update the Q_map
                Q_map[A, S[0], S[1]] = Q_map[A, S[0], S[1]] + \
                            self.alpha * (R + (self.gamma * Q_map[A_prime, S_prime[0], S_prime[1]]) - Q_map[A, S[0], S[1]])

                # update S and A
                S = S_prime
                A = A_prime 

            reward_tracker.append(idx)

        mean_q_map: np.array = np.mean(Q_map, axis=0)
        print(f'Stacked Q_map: \n {mean_q_map}')

        # plot reward
        # plt.plot(reward_tracker)
        # plt.show()

        figure(figsize=(14, 6), dpi=80)
        plt.plot(reward_tracker, label = "Penalty", linestyle="-", color='blue', linewidth=2.5)
        plt.title(f'Pentaly Over Traing Episodes, Bug Spread Coefficient: {MDP.wildfire_update_window}', fontsize=20)
        plt.xlabel('Episode Number', fontsize=14)
        plt.ylabel('Penalty for Episode (1/Final Reward)', fontsize=14)
        plt.legend()
        plt.show()

        if MDP.generate_plots: 
            sns.heatmap(mean_q_map, annot=True, linewidth=.5, cmap="crest")
            # TODO add title with num training episodes 
            plt.show()

        if MDP.save_policy:
            fullpath = PATH_TO_POLICIES / 'Q_map.npy'
            np.save(str(fullpath), Q_map)
            print(f'SARSA Policy saved to the following location: \n {fullpath}')


    def generate_video(self):
        """
        Function used to create and save an MP4 file showing the agent's path using the loaded policy
        """
        print('Generating MP4 file - be patient, this can take a while...')
        # load the saved policy 
        fullpath = PATH_TO_POLICIES / 'Q_map.npy'
        Q_map = np.load(str(fullpath))

        # load the board from a boards.py file 
        board = self.board_obj.original_board
        problem_name = 'MDP BUG CONTAINMENT'
        agent_start_position = (3,3)
        fire_start_position = (0,0)
        # create an environment 
        solution = Solution(problem_name=problem_name, model_name='SARSA')
        env = MDPStaticEnv(board=board, agent_start_position=agent_start_position, solution=solution, 
                        fire_start_position=fire_start_position, action_space=self.action_space, 
                        VERBOSE=False)
        # initialize the agent(s) and the fire start locations
        env.initialize_state()
        S = env.agent_position
        # get max action because it's inference time
        A = self.get_max_axtion(Q_map=Q_map, S=S, action_space=self.action_space)
        env_snapshots = []  # <-- used for generating mp4
        while env.solution.solved is False: 
            env.increment_time(action=A)
            # define the reward and the new state
            R = env.calculate_reward(action=A)  # <-- we need to update this so it will work for SARSA
            S_prime = env.agent_position
            # choose the next action (from S_prime) based on Q_map
            A_prime = self.get_action(Q_map=Q_map, S=S_prime, action_space=self.action_space)
            # update the Q_map
            Q_map[A, S[0], S[1]] = Q_map[A, S[0], S[1]] + \
                        self.alpha * (R + (self.gamma * Q_map[A_prime, S_prime[0], S_prime[1]]) - Q_map[A, S[0], S[1]])
            # update S and A
            S = S_prime
            A = A_prime 

            big_board1 = env.expand_board(board=env.board)
            big_board2 = env.expand_board(board=(env.agent_board/100))
            for i in range(5):
                env_snapshots.append(big_board1 + big_board2)

        env_itr = iter(env_snapshots)
        # write the animation file 
        now = datetime.datetime.now()
        timestamp = now.strftime("%H-%M-%S")
        mov_fname: str = f'static-env-{timestamp}.mp4'
        MOV_PATH = PATH_TO_MP4S / mov_fname
        write_animation(itr=env_itr, out_file=MOV_PATH)
    

def main():

    # define the SARSA model with all of the necessary environment parameters 
    clf = TDLearning(problem_name='Contain the Bug!', 
                board_obj = board2, 
                action_space = [0, 1, 2, 3, 4], 
                plot_q_map=MDP.generate_plots,
                alpha = 0.3,
                gamma = 0.9, 
                epsilon = 0.5, 
                n_episodes = MDP.episodes)
    
    # test SARSA
    clf.sarsa()

    if MDP.generate_mp4:
        clf.generate_video()

    # test Q-learning 
    # clf.q_learning()

if __name__ == "__main__":
    main()