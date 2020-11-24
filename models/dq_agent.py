import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import time


class NeuralNetwork(nn.Module):
    def __init__(self, lr, inp_dim, l1_dim, l2_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(*inp_dim, l1_dim)
        self.layer2 = nn.Linear(l1_dim, l2_dim)
        self.output = nn.Linear(l2_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = t.device('cuda:0')
        self.to(self.device)
    
    def forward(self, state):
        x = self.layer1(state)
        x = f.relu(x)
        x = self.layer2(x)
        x = f.relu(x)
        
        out = self.output(x)
        
        return out
    
class Agent:
    def __init__(self, inp_dim, out_dim, gamma, lr, batch_size, mem_size):
        self.nn = NeuralNetwork(lr, inp_dim, 256, 256, out_dim)
        
        self.states = t.zeros((mem_size, inp_dim[0]), dtype=t.float32)
        self.states_batch = t.zeros((batch_size, inp_dim[0]), dtype=t.float32)
        
        self.actions = t.zeros((mem_size, out_dim), dtype=t.float32)
        self.actions_batch = t.zeros((batch_size, out_dim), dtype=t.float32)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_size = mem_size
        
        self.learned = 0
        self.batch_indx = 0
        
    def store(self, state, action, reward):
        indx = self.learned
        if indx >= self.mem_size:
            indx = indx % self.mem_size
        self.states[indx] = t.tensor(state)
        self.actions[indx] = t.zeros(self.actions.shape[1])
        self.actions[indx][action] = reward
        self.learned += 1
        
    def learn(self):
        batch_indx = self.batch_indx
        bound = min(self.mem_size, self.learned)
        exp_indx = np.random.choice(bound)
        
        self.states_batch[batch_indx] = self.states[exp_indx]
        self.actions_batch[batch_indx] = self.actions[exp_indx]
        
        self.batch_indx = (batch_indx + 1) % self.batch_size
        
        if self.batch_indx < self.learned:
            return
        
        self.nn.optimizer.zero_grad()
        states = self.states_batch.to(self.nn.device)
        q_eval = self.nn.forward(states).to(self.nn.device)
        q_target = self.actions_batch.to(self.nn.device)
        
        
        loss = self.nn.loss(q_target, q_eval).to(self.nn.device)
        loss.backward()
        self.nn.optimizer.step()
        
        
    def predict(self, state, randomness=0):
        rand = np.random.random()
        if rand > randomness:
            state = t.tensor([state]).to(self.nn.device)
            probabilities = self.nn.forward(state)
            action = t.argmax(probabilities).item()
            return action
        else:
            return np.random.randint(4)
    
    def load_nn(self, path):
        self.nn.load_state_dict(t.load(path))
        self.nn.eval()
        
    def save_nn(self, path):
        t.save(self.nn.state_dict(), path)
        