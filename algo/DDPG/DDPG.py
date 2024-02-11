import copy
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space = None):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) /2.).to(device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) /2.).to(device)
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action =  self.action_scale* torch.tanh(self.l3(x)) + self.action_bias
        return action
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat((state,action),1)))
        x = F.relu(self.l2(x))
        return self.l3(x)
    
class DDPG(object):
    def __init__(self,state_dim, action_dim, action_space, tau=0.005, discount=0.99, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, action_space).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.tau = tau
        self.discount = discount

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        entropy = np.array([0])
        return self.actor(state).cpu().data.numpy().flatten(), entropy, self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            target_q = reward + not_done * self.discount * self.critic_target(next_state, self.actor_target(next_state))

        critic_loss = F.mse_loss(target_q, self.critic(state, action))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - (self.critic(state, self.actor(state))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename  + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
