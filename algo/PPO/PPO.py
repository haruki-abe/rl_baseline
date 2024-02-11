import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l_mean = nn.Linear(64,action_dim)
        self.l_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action_mean = torch.tanh(self.l_mean(x))
        return action_mean, self.l_logstd
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        v = self.l3(x)
        return v

class PPO(object):
    def __init__(self, state_dim, action_dim ,discount, lr, update_epochs, clip_coef, target_kl, norm_adv, ent_coef, max_grad_norm):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr)
        self.value = Critic(state_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(),lr=lr)
        self.rollout_buffer = RolloutBuffer()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.max_grad_norm = max_grad_norm
        self.discount = discount
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action_mean, action_logstd = self.actor(state)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean,action_std)
        action = dist.sample()
        return action.cpu().data.numpy().flatten()
    
    def get_action_value(self, state, action=None):
        state = torch.FloatTensor(state).to(device)
        action_mean, action_logstd = self.actor(state)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean,action_std)
        if action==None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.value(state)
    
    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.value(state)

    def train(self, batch_size, advantages, returns):
        # states = self.rollout_buffer.states.reshape((-1,) + (self.state_dim,))
        # logprobs = self.rollout_buffer.logprobs.reshape(-1)
        # actions = self.rollout_buffer.actions.reshape((-1,) + (self.action_dim,))
        # advantages = advantages.reshape(-1)
        # returns = returns.reshape(-1)
        # values = self.rollout_buffer.state_values.reshape(-1)
        states = torch.stack(self.rollout_buffer.states).reshape((-1,) + (self.state_dim,)).to(device)
        logprobs = torch.stack(self.rollout_buffer.logprobs).to(device).reshape(-1)
        actions = torch.stack(self.rollout_buffer.actions).reshape((-1,) + (self.action_dim,)).to(device)
        advantages = advantages.to(device).reshape(-1)
        returns = returns.to(device).reshape(-1)
        values = torch.stack(self.rollout_buffer.state_values).to(device).reshape(-1)

        inds = np.arange(returns.shape[0])

        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, returns.shape[0], batch_size):
                end = start + batch_size
                b_ind = inds[start:end]

                _, newlogprob, entorpy, newvalue = self.get_action_value(states[b_ind].cpu().data.numpy(), actions[b_ind])
                logratio = newlogprob - logprobs[b_ind]
                ratio = logratio.exp()


                with torch.no_grad():
                    # http://joschu.net/blog/kl-approx.html の方法でklを近似
                    approx_kl = ((ratio - 1) - logratio).mean()
                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

                b_advantages = advantages[b_ind]

                if self.norm_adv:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                policy_loss1 = -b_advantages * ratio
                policy_loss2 = -b_advantages * torch.clamp(ratio, 1-self.clip_coef,1+self.clip_coef)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                value_loss = F.mse_loss(newvalue, returns[b_ind])

                entorpy_loss = entorpy.mean()

                policy_loss += entorpy_loss * self.ent_coef

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm) #勾配をクリッピング
                self.actor_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

            else:
                continue
            break
    

