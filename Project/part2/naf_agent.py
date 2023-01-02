#from distutils.command.config import config
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
sys.path.insert(0, os.path.abspath("../.."))
from common import helper as h
from torch.distributions import MultivariateNormal

#device = torch.device('cuda')
device = torch.device('cpu')
"""
NAF has as base network four hidden layers, each of size 256, with skip connections from input state and batch normalization between layers. The 3 output heads are:
    - mu, outputs action with the highest value
    - P : outputs semidefinite covariance matrix
    - V : outputs value for state
    
Advantage(action) = (action - mu).T @ P @ (action - mu)
Q(state, action) = V(state) + Advantage(action)
"""
class NAF(nn.Module):
    def __init__(self, state_shape, action_size,
                 batch_size=256, hidden_dims=256, gamma=0.99, lr=1e-3, grad_clip_norm=1.0, tau=0.005):
        super(NAF, self).__init__()
        # base layers 1-4 with skip connections
        # layer 1
        self.layeri1 = nn.Linear(in_features = state_shape, out_features = hidden_dims)
        self.batchn1 = nn.BatchNorm1d(hidden_dims)
        # layer 2
        self.layeri2 = nn.Linear(hidden_dims, hidden_dims) 
        self.batchn2 = nn.BatchNorm1d(hidden_dims)   
        # layer 1-4 are shared with all the heads
        # heads:
        # mu : maximum action, apply Tanh in forward
        self.mu = nn.Linear(hidden_dims, action_size)
        # lower triangular entries of L, apply Tanh in forward
        self.L_entries = nn.Linear(hidden_dims, int(action_size * (action_size + 1) / 2))
        # value function (scalar)
        self.V = nn.Linear(hidden_dims, 1)
        # store constructor parameters   
        self.action_size = action_size
        self.state_dim = state_shape
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.counter = 0
        
    def forward(self, state, action=None, min_action=-1, max_action=1):
        # Initialize as none by default
        Advantage = None
        Q = None 
        
        # base layers feedforward
        #x = self.bn0(state) # should we normalize before first layer?
        x = torch.relu(self.layeri1(state))
        x = self.batchn1(x)
        x = torch.relu(self.layeri2(x))
        x = self.batchn2(x)
        # maximum action
        mu = torch.tanh(self.mu(x))
        max_action_value = mu.unsqueeze(-1)
        l_entries = torch.tanh(self.L_entries(x)).to(device)
        Value = self.V(x)
        # create lower triangular matrix L
        #L = torch.zeros((self.state_dim, self.action_size, self.action_size))
        L = torch.zeros((state.shape[0], self.action_size, self.action_size)).to(device)
        indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0).to(device)
        L[:, indices[0], indices[1]] = l_entries
        L.diagonal(dim1=1,dim2=2).exp_() # exponentiate diagonal elements
        # Taken from forum. TODO: think about how does this work?
        P = L * L.transpose(2, 1) 
        # But not this?
        #P = L @ L.transpose(1, 2)        
        if action!=None:
            Advantage = (-0.5 * (action.unsqueeze(-1) - max_action_value).transpose(2, 1) @ P @ (action.unsqueeze(-1) - max_action_value)).squeeze(-1)
            Q = Advantage + Value
        
        # Add some noise to actioon
        expl_noise = 0.3 * max_action
        action = mu + (expl_noise * torch.randn(self.action_size)).to(device)
          
        return action, Q, Value, mu.squeeze(-1)

# Modified from exercise 4
class NAFAgent(object):
    def __init__(self, state_shape, n_actions,
                 batch_size=256, hidden_dims=256, gamma=0.99, lr=1e-3, grad_clip_norm=1.0, tau=0.005, use_shallow=False):
        self.n_actions = n_actions
        self.state_dim = state_shape[0]
        self.policy_net = NAF(self.state_dim, n_actions, hidden_dims, 1024).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.counter = 0
    
    def update(self, buffer):
        batch = buffer.sample(self.batch_size, device=device)
        # take values from BUfferi
        states, actions, rewards, next_states, not_dones = batch.state, batch.action, batch.reward, batch.next_state, batch.not_done, 
        # move the buffer elements to device in float32 format
        states = torch.tensor(states).to(device)
        next_states =torch.tensor(next_states).to(device)
        rewards = torch.tensor(rewards).to(device)
        not_dones = torch.tensor(not_dones).to(device)
        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, Vprime, _ = self.target_net(next_states)

        # Compute Q targets for current states 
        V_targets = rewards + (self.gamma * Vprime * not_dones)
        
        # Get expected Q values from local model
        _, Q, _, _ = self.policy_net(states, actions)

        # Compute loss
        loss = F.mse_loss(Q, V_targets) 
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        h.soft_update_params(self.policy_net, self.target_net, self.tau)
            
        return {'loss': loss.item(),
                 'num_update': 1}
        
    @torch.no_grad()
    def get_action(self, state):
        self.policy_net.eval()
        if state.ndim == 1:
            state = state[None]  # add the batch dimension
        x = torch.from_numpy(state).float().to(device)
        action, _,_,_ = self.policy_net(x)
        self.policy_net.train()
        return action
    
    @torch.no_grad()
    def get_action_wo_noise(self, state):
        self.policy_net.eval()
        if state.ndim == 1:
            state = state[None]  # add the batch dimension
        x = torch.from_numpy(state).float().to(device)
        _,_,_, action_value = self.policy_net(x)
        self.policy_net.train()
        return action_value.squeeze(-1)


    def save(self, fp):
        path = fp/'naf.pt'
        torch.save({
            'policy': self.policy_net.state_dict(),
            'policy_target': self.target_net.state_dict()
        }, path)

    def load(self, fp):

        path = fp/'naf.pt'
        d = torch.load(path)
        self.policy_net.load_state_dict(d['policy'])
        self.target_net.load_state_dict(d['policy_target'])