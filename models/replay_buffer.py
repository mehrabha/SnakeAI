import torch as t
import numpy as np


class ReplayBuffer():
    def __init__(self, inp_dim, mem_size, batch_size):
        self.device = t.device('cuda:0')
        self.states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.new_states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.terminal = np.zeros(mem_size, dtype=np.bool)
        self.priorities = np.zeros(mem_size, dtype=np.float32)
        
        self.states_batch = np.zeros((batch_size, inp_dim[0]), dtype=np.float32)
        self.actions_batch = np.zeros(batch_size, dtype=np.int32)
        self.rewards_batch = np.zeros(batch_size, dtype=np.float32)
        self.new_states_batch = np.zeros((batch_size, inp_dim[0]), dtype=np.float32)
        self.terminal_batch = np.zeros(batch_size, dtype=np.bool)
        
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_index = 0
        self.batch_full = False
    
    def add(self, state, action, reward, new_state, terminated, error):
        indx = self.mem_index
        self.states[indx] = state
        self.new_states[indx] = new_state
        self.actions[indx] = action
        self.rewards[indx] = reward
        self.terminal[indx] = terminated
        self.priorities[indx] = error
        
        if self.mem_index >= self.batch_size:
            self.batch_full = True
            
        self.mem_index += 1
        if self.mem_index >= self.mem_size:
            self.mem_index = 0
            self.mem_full = True
    
    def get_probabilities(self, scale_factor):
        priorities = t.from_numpy(self.priorities).to(self.device)
        scaled_priorities = t.pow(priorities, scale_factor).to(self.device)
        sample_probabilities = t.div(scaled_priorities, 
                                     t.sum(scaled_priorities).item())
        return sample_probabilities.numpy()
    
    def sample(self, priority_scale=1.0):
        bound = self.mem_size if self.mem_full else self.mem_index
        
        probs = self.get_probabilities(priority_scale)
        indx = np.random.choice(bound, self.batch_size, probs)
        
        return (self.states[indx], self.new_states[indx], self.actions[indx],
                self.rewards[indx], self.terminal[indx])
    
    def is_batch_full(self):
        return self.batch_full

