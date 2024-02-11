import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space = None):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,action_dim)
        self.l4 = nn.Linear(256,action_dim)

        # actionが正規化されているかどうか．Samplingのあとでの処理が異なる．TD3のmax_actionと同じ
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(device)


    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.l3(x)
        log_std = self.l4(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std
    
    def sample(self,state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        xs = dist.rsample()
        ys = torch.tanh(xs)
        action = ys * self.action_scale + self.action_bias
        log_prob =  dist.log_prob(xs)
        log_prob = log_prob - torch.log(self.action_scale * (1 - ys.pow(2)) + epsilon)
        log_prob = log_prob.sum(1,keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256,1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256,1)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat([state, action],1)))
        x = F.relu(self.l2(x))
        q1 = self.l3(x)

        x = F.relu(self.l4(torch.cat([state, action], 1)))
        x = F.relu(self.l5(x))
        q2 = self.l6(x)

        return q1, q2

class SAC(object):
    def __init__(self, state_dim, action_dim, action_space=None,lr=3e-4, discount=0.99,tau=0.005, alpha =0.2, alpha_tuning = True):
        self.actor = Actor(state_dim, action_dim, action_space).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.alpha_tuing = alpha_tuning

        if alpha_tuning:
            self.alpha_target = - torch.prod(torch.Tensor(action_space.shape)).to(device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action, log_prob, mean = self.actor.sample(state)
        entropy = - log_prob
        return action.cpu().data.numpy().flatten(), entropy.cpu().data.numpy().flatten(), mean.cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size = 256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            next_action, log_prob, mean = self.actor.sample(next_state)
            q1, q2 = self.critic_target(next_state, next_action)
            minq = torch.min(q1,q2)
            y = reward + self.discount * not_done * (minq - self.alpha * log_prob)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        action, log_prob, mean = self.actor.sample(state)
        q1, q2 = self.critic(state, action)
        actor_loss = -(torch.min(q1,q2) - self.alpha * log_prob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.alpha_tuing:
            with torch.no_grad():
                _, log_prob, _ = self.actor.sample(state)
            alpha_loss = -(self.log_alpha.exp() * (log_prob + self.alpha_target).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlog = self.alpha.clone()

        else:
            alpha_loss = torch.tensor(0.)
            alpha_tlog = torch.tensor(self.alpha)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlog.item()


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        

        
