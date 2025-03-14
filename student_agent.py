# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

from ext import ExternalAPI
import torch

ext2 = None
q_table = None

def get_action(obs):
    global ext2, q_table
    if ext2 == None:
        ext2 = ExternalAPI()
    if q_table == None:
        with open("q_table.pkl", mode="rb") as file:
            q_table = pickle.load(file)
    state = ext2.get_state(obs)
    # action = q_table.get_action(state, 0)
    action = np.argmax(q_table[state[0]][state[1][0]][state[1][1]][state[2][0]][state[2][1]][state[2][2]][state[2][3]])
    print(action)
    ext2.add_action(action)
    return action

# get_action((2, 2, 0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0))
