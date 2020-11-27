from models.neural_nets import NeuralNetwork
import torch as t
import numpy as np

class Agent:
    def __init__(self, inp_dim, out_dim, hidden_dims=128, gamma=.99, 
                 lr=.03, batch_size=256, mem_size=100000):
        self.nn = NeuralNetwork(lr, inp_dim, hidden_dims, out_dim)
        self.states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.new_states = np.zeros((mem_size, inp_dim[0]), dtype=np.float32)
        self.terminal = np.zeros(mem_size, dtype=np.bool)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_size = mem_size
        
        self.learned = 0
        
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
        bound = min(self.mem_size, self.learned)
        
        if bound < self.batch_size:
            return  
        
        self.nn.optimizer.zero_grad()
        batch = np.random.choice(bound, self.batch_size, replace=False)
        
        states_batch = t.tensor(self.states[batch]).to(self.nn.device)
        actions_batch = self.actions[batch]
        rewards_batch = t.tensor(self.rewards[batch]).to(self.nn.device)
        new_states_batch = t.tensor(self.new_states[batch]).to(self.nn.device)
        terminal_batch = t.tensor(self.terminal[batch]).to(self.nn.device)
        
        indexes = t.arange(self.batch_size).to(self.nn.device)
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
        