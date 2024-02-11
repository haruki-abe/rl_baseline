import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, action_space) -> None:
        super(Actor, self).__init__()

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,action_dim)

        self.action_scale = torch.tensor((action_space.high - action_space.low) /2., requires_grad=False).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) /2., requires_grad=False).to(device)
        print(self.action_scale)
        print(self.action_bias)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        action = self.action_scale* x + self.action_bias
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        #Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)

        #Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256,256)
        self.l6 = nn.Linear(256,1)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat((state, action),1)))
        x = F.relu(self.l2(x))
        value_1 = self.l3(x)

        x = F.relu(self.l4(torch.cat((state, action),1)))
        x = F.relu(self.l5(x))
        value_2 = self.l6(x)

        return value_1, value_2
    
    def q1(self, state, action):
        x = F.relu(self.l1(torch.cat((state, action),1)))
        x = F.relu(self.l2(x))
        value_1 = self.l3(x)

        return value_1
    
class TD3():
    def __init__(self,action_dim,state_dim,action_space,policy_noise=0.2,noise_clip=0.5,discount=0.99,delay=2,tau=0.005,lr=3e-4) -> None:
        self.actor = Actor(state_dim,action_dim,action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.discount = discount
        self.step = 0
        self.delay = delay
        self.tau = tau

        self.action_high =torch.tensor(action_space.high,requires_grad=False).to(device)
        self.action_low =torch.tensor(action_space.low, requires_grad=False).to(device)
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action = self.actor(state)
        entropy = np.array([0])
        return action.cpu().data.numpy().flatten(), entropy, action.cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        self.step += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip,self.noise_clip)*self.actor_target.action_scale

            next_action = (self.actor_target(next_state) + noise).clamp(self.action_low,self.action_high)
            q1, q2 = self.critic_target(next_state,next_action)
            y = reward + not_done * self.discount * torch.min(q1, q2)

        current_Q1, current_Q2 = self.critic(state,action)
        
        critic_loss = F.mse_loss(current_Q1,y) + F.mse_loss(current_Q2,y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.step % self.delay == 0:
            actor_loss = - self.critic.q1(state,self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self,filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
