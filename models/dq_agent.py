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
    def __init__(self, inp_dim, out_dim, gamma=.99, 
                 lr=.03, batch_size=256, mem_size=100000):
        self.nn = NeuralNetwork(lr, inp_dim, 256, 256, out_dim)
        
        self.states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.states_batch = np.zeros((batch_size, inp_dim[0]), dtype=np.float32)
        
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.actions_batch = np.zeros(batch_size, dtype=np.int32)
        
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.rewards_batch = np.zeros(batch_size, dtype=np.float32)
        
        self.new_states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.new_states_batch = np.zeros((batch_size, inp_dim[0]), dtype=np.float32)
        
        self.terminal = np.zeros(mem_size, dtype=np.bool)
        self.terminal_batch = np.zeros(batch_size, dtype=np.bool)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_size = mem_size
        
        self.learned = 0
        self.batch_indx = 0
        
    def store(self, state, action, reward, new_state, terminated):
        indx = self.learned
        if indx >= self.mem_size:
            indx = indx % self.mem_size
            
        self.states[indx] = state
        self.new_states[indx] = new_state
        self.actions[indx] = action
        self.rewards[indx] = reward
        self.terminal[indx] = terminated
        
        self.learned += 1
        
    def learn(self):
        batch_indx = self.batch_indx
        bound = min(self.mem_size, self.learned)
        exp_indx = np.random.choice(bound)
        
        self.states_batch[batch_indx] = self.states[exp_indx]
        self.actions_batch[batch_indx] = self.actions[exp_indx]
        self.rewards_batch[batch_indx] = self.rewards[exp_indx]
        self.new_states_batch[batch_indx] = self.new_states[exp_indx]
        self.terminal_batch[batch_indx] = self.terminal[exp_indx]
        
        self.batch_indx = (batch_indx + 1) % self.batch_size
        
        if self.learned < self.batch_size:
            return
        
        self.nn.optimizer.zero_grad()
        
        states = t.tensor(self.states[:bound]).to(self.nn.device)
        new_states = t.tensor(self.new_states[:bound]).to(self.nn.device)
        rewards = t.tensor(self.rewards[:bound]).to(self.nn.device)
        terminal = t.tensor(self.terminal[:bound]).to(self.nn.device)
        indexes = np.arange(bound)
        
        q_eval = self.nn.forward(states)[indexes, self.actions[:bound]]
        q_next = self.nn.forward(new_states)
        q_next[terminal] = 0.0
        q_target = rewards #+ self.gamma * t.max(q_next, dim=1)[0]
        
        
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
        