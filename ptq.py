import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PytorchQTable:
    def __init__(self, lr):
        self.state_size = (2, 19, 19, 2, 2, 2, 2)
        self.action_size = 6
        self.q_table = nn.Parameter(torch.zeros(*self.state_size, self.action_size, device=device))
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.SGD([self.q_table], lr=lr)
    
    def convert_state(self, state):
        return (state[0], state[1][0] + 9, state[1][1] + 9, state[2][0], state[2][1], state[2][2], state[2][3])

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        return torch.argmax(self.q_table[self.convert_state(state)]).item()
    
    def update(self, state, action, target):
        state_here = self.convert_state(state)
        self.optimizer.zero_grad()
        print(state_here)
        current_q = self.q_table[state_here[0], state_here[1], state_here[2], state_here[3], state_here[4], state_here[5], state_here[6], action]
        # current_q = self.q_table[*state_here, action]
        loss = self.loss_func(current_q, torch.tensor(target).to(device=device))
        loss.backward()
        self.optimizer.step()

