import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 512):
        super().__init__()
        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state,action],1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        q1 = self.l3(x)

        x = torch.cat([state, action], 1)
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        q2 = self.l6(x)

        return q1, q2
    
class CEM():
    def __init__(self,action_dim, action_scale=1.0, action_bias=0.0):
        self.action_dim = action_dim
        self.action_bias = action_bias
        self.action_scale = action_scale

        self.mean = self.action_bias * np.ones(self.action_dim)
        self.std = self.action_scale * np.ones(self.action_dim)

    def initialize(self):
        self.mean = self.action_bias * np.ones(self.action_dim)
        self.std = self.action_scale * np.ones(self.action_dim)

    def sample(self, n):
        actions = self.mean + np.random.normal(size=self.action_dim * n).reshape(n,self.action_dim) * self.action_scale
        return actions
    
    def update(self, samples):
        self.mean = np.mean(samples,axis=0)
        self.std = np.std(samples, axis=0)

 
    
class QT_Opt(object):
    def __init__(self, state_dim, action_dim,action_space, discount=0.99, tau=0.005, lr=3e-4, cem_iter=2, select_num=5, num_samples=64):
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.discount = discount
        self.tau = tau

        self.cem_iter = cem_iter
        self.select_num =select_num
        self.num_samples = num_samples

        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.CEM = CEM(action_dim, np.array((action_space.high - action_space.low) / 2.),np.array((action_space.high + action_space.low) / 2.))


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        self.CEM.initialize()
        for i in range(self.cem_iter):
            actions = self.CEM.sample(self.select_num)
            q1, q2 = self.critic(torch.tile(state,(self.select_num,1)),torch.FloatTensor(actions).to(device))
            qs = torch.min(q1,q2).cpu().data.numpy().flatten()
            idx = qs.argsort()[-int(self.select_num):]
            max_idx = qs.argsort()[-1]
            self.CEM.update(actions[idx])

        action = actions[max_idx]
        return action.flatten()

    def train(self, replay_buffer, batch_size = 512):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        next_state_ = next_state.cpu().data.numpy()
        next_action_list = []
        for i in range(batch_size):         
            next_action_list.append(self.select_action(next_state_[i]))

        next_action = torch.FloatTensor(np.array(next_action_list)).to(device)

        with torch.no_grad():
            q1, q2 = self.critic_target(next_state, next_action)
            y = reward + self.discount * not_done * torch.min(q1,q2)    

        q1, q2 = self.critic(state,action)
        critic_loss = F.mse_loss(y, q1) + F.mse_loss(y, q2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param)
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

    





