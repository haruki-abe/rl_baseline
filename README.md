# Reinforcement Learning Algorithms

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3), Deep Deterministic Policy Gradient (DDPG), Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO) and QT-Opt.

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 2.2](https://github.com/pytorch/pytorch) and Python 3.7. 

### Usage
Experiments on single environments can be run by calling:
```
python algo/SAC/main_SAC.py --env HalfCheetah-v2
```

Hyper-parameters can be modified with different arguments to main.py. 



