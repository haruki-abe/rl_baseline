# 🚀 Reinforcement Learning Algorithms in PyTorch

A clean and modular PyTorch implementation of popular deep reinforcement learning algorithms for continuous control:

- **TD3** – Twin Delayed Deep Deterministic Policy Gradient  
- **DDPG** – Deep Deterministic Policy Gradient  
- **SAC** – Soft Actor-Critic  
- **PPO** – Proximal Policy Optimization  
- **QT-Opt** – Q-learning with continuous actions (QT-Opt)

These algorithms have been benchmarked on classic **MuJoCo** tasks via **OpenAI Gym** environments.

---

## 🔧 Tech Stack

- **Framework**: [PyTorch 2.2](https://github.com/pytorch/pytorch)  
- **Language**: Python 3.7  
- **Environments**: [MuJoCo](http://www.mujoco.org/), [OpenAI Gym](https://github.com/openai/gym)

---

## 🕹️ Getting Started

To run experiments on a specific environment, simply use:

```bash
python algo/SAC/main_SAC.py --env HalfCheetah-v2
```

🧠 Customize Your Training
All key hyperparameters are exposed for easy experimentation — perfect for both research and hobby projects.

Example custom run:
```bash
python algo/TD3/main_TD3.py --env Walker2d-v2 --seed 42 --start_timesteps 10000 --expl_noise 0.1
```

