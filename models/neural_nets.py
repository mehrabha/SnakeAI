import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, lr, inp_dim, l1_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(*inp_dim, l1_dim)
        #self.layer2 = nn.Linear(l1_dim, l2_dim)
        self.output = nn.Linear(l1_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = t.device('cuda:0')
        self.to(self.device)
    
    def forward(self, state):
        x = self.layer1(state)
        x = f.relu(x)
        #x = self.layer2(x)
        #x = f.relu(x)
        
        out = self.output(x)
        
        return out