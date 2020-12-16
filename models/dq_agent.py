import torch as t
import numpy as np
    

class Agent:
    def __init__(self, nn, inp_dim, out_dim, memory=None, gamma=.99):
        self.nn = nn
        self.gamma = gamma
        self.memory = memory
        self.learned = 0
        
    def store(self, state, action, reward, new_state, terminated):
        # Add experience to buffer
        self.memory.add(state, action, reward, new_state, terminated)
        self.learned += 1
    
    def learn(self, n_batches=1):
        if not self.memory.is_mem_full():
            return
        
        # Create n batches (size=batch_size) and learn
        for i in range(n_batches):
            self.nn.optimizer.zero_grad()
            sample = self.memory.sample()
            
            states_batch = t.tensor(sample[0]).to(self.nn.device)
            new_states_batch = t.tensor(sample[1]).to(self.nn.device)
            actions_batch = sample[2]
            rewards_batch = t.tensor(sample[3]).to(self.nn.device)
            terminal_batch = t.tensor(sample[4]).to(self.nn.device)
            
            indexes = t.arange(actions_batch.size)
            q_eval = self.nn.forward(states_batch)[indexes, actions_batch]
            q_next = t.max(self.nn.forward(new_states_batch), dim=1)[0]
            q_next[terminal_batch] = 0.0
            q_target = rewards_batch + self.gamma * q_next
            
            loss = self.nn.loss(q_target, q_eval).to(self.nn.device)
            loss.backward()
            self.nn.optimizer.step()
        
        
    def predict(self, state, randomness=0):
        rand = np.random.random()
        if rand > randomness:
            state = t.tensor([state]).to(self.nn.device)
            probabilities = self.nn.forward(state)
            #print(probabilities)
            action = t.argmax(probabilities[0]).item()
            return action
        else:
            return np.random.randint(4)
    
    def load_nn(self, path):
        self.nn.load_state_dict(t.load(path))
        self.nn.eval()
        
    def save_nn(self, path):
        t.save(self.nn.state_dict(), path)
        