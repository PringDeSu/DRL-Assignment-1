# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

import pickle
from ext import ExternalAPI

ext2 = ExternalAPI()

loaded = False
q_table = None

def get_action(obs):
    global ext2, loaded, q_table
    if not loaded:
        with open("q_table.pkl", mode="rb") as file:
            q_table = pickle.load(file)
        loaded = True

    state = ext2.get_state(obs)
    action = None
    if not (state in q_table):
        action = np.random.randint(4)
        while state[2][action] == 1:
            action = np.random.randint(4)
    else:
        action = q_table[state].argmax()
    ext2.add_action(action)
    return action

# def get_action(obs):
#
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.
#
#
#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

