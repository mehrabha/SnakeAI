import numpy as np


class ReplayBuffer():
    def __init__(self, inp_dim, mem_size, batch_size, priority_scale=1):
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
        self.priority_scale = priority_scale
        self.mem_index = 0
        self.priority_sum = 0
        self.mem_full = False
    
    def add(self, state, action, reward, new_state, terminated):
        indx = self.mem_index
        self.states[indx] = state
        self.new_states[indx] = new_state
        self.actions[indx] = action
        self.rewards[indx] = reward
        self.terminal[indx] = terminated
        
        self.priority_sum -= self.priorities[indx]
        self.priority_sum += self.priority_scale if terminated else 1
        self.priorities[indx] = self.priority_scale if terminated else 1
            
        self.mem_index += 1
        if self.mem_index >= self.mem_size:
            self.mem_index = 0
            self.mem_full = True
    
    def sample(self):
        if self.batch_size == self.rewards.shape:
            return (self.states, self.new_states, self.actions,
                    self.rewards, self.terminal)
        indx = None
        if self.priority_scale > 1:
            # Generate probabilities of samples
            probs = self.priorities / self.priority_sum
            
            # Get indexes of choices
            indx = np.random.choice(self.mem_size, self.batch_size, 
                                    p=probs, replace=False)
        else:
            indx = np.random.choice(self.mem_size, self.batch_size, 
                                    replace=False)
        
        return (self.states[indx], self.new_states[indx], self.actions[indx],
                self.rewards[indx], self.terminal[indx])
    
    def is_mem_full(self):
        return self.mem_full

