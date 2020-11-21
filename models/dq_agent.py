import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import time


class NeuralNetwork(nn.Module):
    def __init__(self, inp_dim, l1_dim, l2_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(*inp_dim, l1_dim)
        self.layer2 = nn.Linear(l1_dim, l2_dim)
        self.output = nn.Linear(l2_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=.05)
        self.loss = nn.MSELoss()
        self.device = t.device('cuda:0')
        self.to(self.device)
    
    def forward(self, state):
        inp = t.tensor(state).to(self.device)
        x = self.layer1(inp)
        x = f.relu(x)
        x = self.layer2(x)
        x = f.relu(x)
        
        out = self.output(x)
        
        return out
    
class Agent:
    def __init__(self, inp_dim, out_dim, mem_size):
        self.nn = NeuralNetwork(inp_dim, 60, 60, out_dim)
        self.mem_size = mem_size
        self.learned = 0
        self.states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.actions = np.zeros((mem_size, out_dim), dtype=np.float32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        
        self.action_space = list(range(out_dim)) # Workaround
        
    def store(self, state, action, reward):
        indx = self.learned
        if indx >= self.mem_size:
            indx = self.mem_size % indx
        self.states[indx] = state
        self.actions[indx][action] = 1.0
        self.rewards[indx] = reward
        self.learned += 1
        
    def learn(self):
        self.nn.optimizer.zero_grad()
        
        bound = min(self.mem_size, self.learned)
        
        states = self.states[: bound]
        rewards = t.tensor(self.rewards[: bound]).to(self.nn.device)
        
        
        q_eval = self.nn.forward(states).to(self.nn.device)
        q_target = self.nn.forward(states).to(self.nn.device)
        
        indexes = np.arange(bound, dtype=np.int32)
        
        actions = self.actions[: bound]
        action_values = np.array(self.action_space, dtype=np.uint8)
        action_indexes = np.dot(actions, action_values)
        q_target[indexes, action_indexes] = rewards
        
        loss = self.nn.loss(q_target, q_eval).to(self.nn.device)
        loss.backward()
        self.nn.optimizer.step()
        
        
    def predict(self, state):
        rand = np.random.random()
        if rand > 0.03:
            probabilities = self.nn.forward(state)
            action = t.argmax(probabilities).item()
            return action
        else:
            return np.random.randint(4)
        