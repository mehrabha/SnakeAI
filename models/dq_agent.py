import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, lr, inp_dim, l1_dim, l2_dim, out_dim):
        super(Neural_Network, self).init()
        
        self.layer1 = nn.Linear(*inp_dim, l1_dim)
        self.layer2 = nn.Linear(l1_dim, l2_dim)
        self.output = nn.Linear(l2_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
    
    def forward(self, state):
        inp = t.Tensor(state)
        
        x = self.layer1(inp)
        x = f.relu(x)
        x = self.layer2(x)
        x = f.relu(x)
        
        out = self.output(x)
        
        return out
    
class Agent:
    def __init__(self, alpha, inp_dim, out_dim, mem_size):
        self.nn = NeuralNetwork(alpha, inp_dim, 72, 72, out_dim)
        self.mem_size = mem_size
        self.learned = 0
        self.states = np.zeros((mem_size, inp_dim), dtype=np.uint8)
        self.actions = np.zeros((mem_size, out_dim), dtype=np.float)
        self.rewards = np.zeros(mem_size, dtype=np.int8)
        
    def store(self, state, action, reward):
        indx = self.mem_size % self.learned
        self.states[indx] = state
        self.actions[indx][action] = 1.0
        self.rewards[indx] = reward
        self.learned += 1
        
    def learn(self):
        self.nn.optimizer.zero_grad()
        
        bound = min(self.mem_size, self.learned)
        
        state = self.states[: bound]
        action = self.actions[: bound]
        reward = self.rewards[: bound]
        #TODO
        
        
        
    def predict(self, state):
        probabilities = self.nn.forward(state)
        action = t.argmax(probabilities).item
        return action
        