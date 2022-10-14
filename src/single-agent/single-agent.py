"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Single Agent RL models

This script uses an environment from settings.py and a model from models.py and 
generates a q-map for the chosen environent using the chosen model. 

This script can be run from the command line using: $ single-agent.py
"""

# local imports 
from settings import StaticEnv
from models import QLearning, SARSA

def main():
    env = StaticEnv.env1

    q_learning = QLearning(env=env)
    q_learning.solve()


if __name__ == "__main__":
    main()
