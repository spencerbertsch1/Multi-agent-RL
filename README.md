# Multi-agent RL

Dartmouth College  
Fall 2022  
Spencer Bertsch, Keshav Inamdar,  Jiahui (Gary) Luo

## To run the code: 

1. Clone [this github repository](https://github.com/spencerbertsch1/Multi-agent-RL) locally and `cd` into the `multi-agent-rl` directory. 
Alternatively, if you acquired this code from a zip file, simply unzip the file locally, `cd` into the `multi-agent-rl` directory, and continue to Step 2. 
In order to run any of the test scripts in this repository, you will need python 3. 
   1. If you don't have python 3 installed, you can install Anaconda and create a new Conda environment using the following command:
      1. `$ conda create -n game_theory_env python=3.9`
   2. Then activate the new environment by running the following command:
       1. `$ conda activate game_theory_env`
   3. At this point you will need to install the requires libraries into you new environment using the following command: 
       1. `$ pip install -r requirements.txt`
   4. Then proceed to the following step. 
   

2. `cd` to the src directory by running the following command:
   1. `$ cd src`

## How to run the single-agent goal seek problem: 

1. `cd` into the `single-agent` directory by running the following command:
   1. `$ cd src`

2. Run `models.py` by running the following command:
   1. `$ python3 models.py`

3. In this case a single agent goal-seek problem will be initiated using SARSA. After 100 training iterations finish, the output policy will be averaged and plotted using seaborn. 

You should see the following output: 

```
SARSA Initiated! Please be patient, training can take a while.
...SARSA 0/100 complete...
...SARSA 10/100 complete...
...SARSA 20/100 complete...
...SARSA 30/100 complete...
...SARSA 40/100 complete...
...SARSA 50/100 complete...
...SARSA 60/100 complete...
...SARSA 70/100 complete...
...SARSA 80/100 complete...
...SARSA 90/100 complete...
Stacked Q_map: 
 [[28.51118287 38.67743831 48.86245208  0.        ]
 [16.22008427  0.38918256 11.26685603  0.        ]
 [11.04076937  3.24245035  2.80816853 -4.35737585]]
```

In addition you should see the following plot: 

 <p align="center">
    <img src="https://github.com/spencerbertsch1/Multi-agent-RL/blob/main/src/diagrams/single_agent.png?raw=true" alt="single_agent" width="60%"/>
</p>

## How to run the single-agent MDP problem: 

1. `cd` into the `mdp` directory by running the following command:
   1. `$ cd mdp`

2. Run `models.py` by running the following command:
   1. `$ python3 models.py`

3. In this case a simple Bug and Farmer game will be initiated and trained using a reward function modeled using simple reward shaping. 

You should see the following output: 

```
SARSA Initiated! Please be patient, training can take a while.
...SARSA 0/50 complete...
...SARSA 10/50 complete...
...SARSA 20/50 complete...
...SARSA 30/50 complete...
...SARSA 40/50 complete...
Stacked Q_map: 
 [[29.04575243 24.23927362 12.82082887  2.62218003]
 [22.67865218 24.93807959 15.57100995  4.89993229]
 [10.83404141 13.28426297  9.81622272  4.73549434]
 [ 1.165653    4.7317372   6.07682824  5.36635993]]
 ```

In addition you should see the following plots showing the final policy after only 50 training episodes and the learning curve showing how much trouble this agent-environment configuration has in converging over only 50 episodes. 

 <p align="center">
    <img src="https://github.com/spencerbertsch1/Multi-agent-RL/blob/main/src/diagrams/mdp.png?raw=true" alt="big graph" width="60%"/>
</p>

 <p align="center">
    <img src="https://github.com/spencerbertsch1/Multi-agent-RL/blob/main/src/diagrams/learning_curve.png?raw=true" alt="big graph" width="60%"/>
</p>