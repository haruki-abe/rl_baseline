import numpy as np
import torch
import gym
import argparse
import os
import random
import os
import sys
from gym.wrappers import RecordVideo
import wandb
import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) 
import utils
import QT_Opt

def eval_policy(policy,env_name,seed,eval_episode,record_video):
    eval_env = gym.make(env_name)
 
    eval_env.seed(seed + 10)

    avg_reward = 0.
    avg_entropy = 0.

    for _ in range(eval_episode):
        state, done = eval_env.reset(), False
        while not done:             
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            
    avg_reward /= eval_episode
    

    print("---------------------------------------")
    print(f"Evaluation over {eval_episode} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    if record_video:
        record_env = gym.make(env_name)
        record_env = RecordVideo(record_env,"./results")
        record_env.seed(seed)
        state, done = record_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = record_env.step(action)
        state, done = record_env.reset(), False

    return avg_reward, avg_entropy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="QT_Opt")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_steps", default=1e6, type=int)
    parser.add_argument("--warmup_steps", default= 25e3, type=int)
    parser.add_argument("--batch_size", default= 256, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--eval_episode", default=10, type=int)
    parser.add_argument("--record_video", default=False, type=bool)
    parser.add_argument("--record_freq", default=5e5, type=int)
    parser.add_argument("--save_freq", default=5e5, type=int)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default= 0.005, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--project", default="comparison_model")
    #QT-Opts
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--cem_iter", default=2, type=int)
    parser.add_argument("--select_num", default=5, type=int)
    parser.add_argument("--num_samples", default=64, type=int)

    

    args = parser.parse_args()

    filename = f"{args.policy}_{args.env}_{args.seed}"

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space
    action_bias = (action_space.high + action_space.low) / 2.
    action_scale = (action_space.high - action_space.low) / 2.

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": action_space,
        "discount": args.discount,
        "tau": args.tau,
        "lr": args.lr
        }
    

    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["noise_clip"] = args.noise_clip
        kwargs["delay"] = args.delay
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    elif args.policy == "SAC":
        kwargs["alpha_tuning"] = args.alpha_tuning
        kwargs["alpha"] = args.alpha
        policy = SAC.SAC(**kwargs)
    elif args.policy == "QT_Opt":
        kwargs["cem_iter"] = args.cem_iter
        kwargs["select_num"] = args.select_num
        kwargs["num_samples"] = args.num_samples
        policy = QT_Opt.QT_Opt(**kwargs)

    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")


    replay_buffer = utils.ReplayBuffer(state_dim=state_dim,action_dim=action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_steps = 0
    episode_num = 0

    wandb.init(
        project=args.project,
        config=vars(args),
        name=filename + datetime.datetime.now().strftime("%m%d%H%M"),
    )

    for i in range(int(args.max_steps)):

        episode_steps += 1

        if i < args.warmup_steps:
            action = env.action_space.sample()

        else:
            if args.policy =="TD3" or args.policy =="DDPG":
                action, _, _ = policy.select_action(np.array(state))
                action = (
                    action
                + np.random.normal(action_bias,  action_scale* args.expl_noise, size = action_dim)
                ).clip(action_space.low, action_space.high)
            elif args.policy =="SAC":
                action, entropy, mean = policy.select_action(np.array(state))
            elif args.policy == "QT_Opt":
                if args.epsilon > random.random():
                    action = policy.select_action(np.array(state))
                else:
                    action = env.action_space.sample()
                
            
        next_state, reward, done, _ = env.step(action)
        #これなんのため？
        done_bool = float(done) if episode_steps < env._max_episode_steps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if i >= args.warmup_steps and replay_buffer.size > args.batch_size:
            policy.train(replay_buffer,args.batch_size)

        if done:
            print(f"Toral T:{i+1} Episode Num:{episode_num} Episode_timesteps:{episode_steps} Reward:{episode_reward:.3f}")
            wandb.log({"train/reward":episode_reward}, step = i+1)
            state, done = env.reset(), False
            episode_num += 1
            episode_steps = 0
            episode_reward = 0

        if (i + 1) % args.eval_freq == 0:
            if (i+1) % args.record_freq == 0 and args.record_video:
                avg_reward, entropy = eval_policy(policy,args.env,args.seed,args.eval_episode,True)
            else:
                avg_reward, entropy = eval_policy(policy,args.env,args.seed,args.eval_episode,False)
            wandb.log({"eval/reward": avg_reward, "eval/entropy": entropy}, step=i+1)
            if (i+1) % args.save_freq == 0:
                policy.save(f"./models/{filename}")
