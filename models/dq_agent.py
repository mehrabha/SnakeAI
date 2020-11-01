import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, lr, inp_dims, l1_dims, l2_dims, out_dims):
        super(Neural_Network, self).init()
        
        self.layer1 = nn.Linear(inp_dims, l1_dims)
        self.layer2 = nn.Linear(l1_dims, l2_dims)
        self.output = nn.Linear(l2_dims, out_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
    
    def forward(self, state):
        inp = t.Tensor(state)
        
        x = self.layer1(inp)
        x = f.relu(x)
        x = self.layer2(x)
        x = f.relu(x)
        
        out = self.output(x)