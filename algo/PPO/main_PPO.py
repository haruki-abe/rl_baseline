import numpy as np
import torch
import gym
import argparse
import os
from gym.wrappers import RecordVideo
import wandb
import datetime

import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_policy(policy,env_name,seed,eval_episode,record_video):
    eval_env = gym.make(env_name)
 
    eval_env.seed(seed + 10)

    avg_reward = 0.

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

    return avg_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="PPO")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_steps", default=1e6, type=int)
    parser.add_argument("--warmup_steps", default= 25e3, type=int)
    parser.add_argument("--batch_size", default= 256, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--eval_episode", default=10, type=int)
    parser.add_argument("--record_video",action="store_true")
    parser.add_argument("--record_freq", default=5e5, type=int)
    parser.add_argument("--save_freq", default=5e5, type=int)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default= 0.005, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--project", default="comparison_model")
    #PPO
    parser.add_argument("--max_ep_len", default=1000, type=int)
    parser.add_argument("--update_epochs", default=4, type=int)
    parser.add_argument("--clip_coef", default=0.2, type=float)
    parser.add_argument("--target_kl", default=0.1, type=float)
    parser.add_argument("--norm_adv", action="store_true")
    parser.add_argument("--ent_coef", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=0.5, type=float)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gae_lambda", default=0.9, type=float)

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
  
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "lr": args.lr
        }

    kwargs["update_epochs"] = args.update_epochs
    kwargs["clip_coef"] = args.clip_coef
    kwargs["target_kl"] = args.target_kl
    kwargs["norm_adv"] = args.norm_adv
    kwargs["ent_coef"] = args.ent_coef
    kwargs["max_grad_norm"] = args.max_grad_norm
    policy = PPO.PPO(**kwargs)

    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")

    wandb.init(
        project=args.project,
        config=vars(args),
        name=filename + datetime.datetime.now().strftime("%m%d%H%M"),
    )

    episode_nums = 0
    step = 0

    while step <= args.max_steps:
        #学習率のアニーリングするなら(未実装)
        if args.anneal_lr:
            frac = 1.0 - (step)/args.max_steps
            lr_now = frac * args.lr
            policy.actor_optimizer.param_groups[0]["lr"] = lr_now
            policy.value_optimizer.param_groups[0]["lr"] = lr_now

        state, done = env.reset(), False
        done_bool = done_bool = float(done)
        episode_reward = 0  
        episode_nums += 1
        for t in range(args.max_ep_len):
            step += 1
            policy.rollout_buffer.is_terminals.insert(t, done_bool)
            policy.rollout_buffer.states.insert(t, torch.tensor(state).to(device)) 
            
            with torch.no_grad():
                action, log_prob, entropy, value = policy.get_action_value(np.array(state).reshape(1,-1)) 
                policy.rollout_buffer.state_values.insert(t, value.flatten())

            policy.rollout_buffer.actions.insert(t, action)
            policy.rollout_buffer.logprobs.insert(t,log_prob)
            next_state, reward, done, _ = env.step(action.cpu().data.numpy().flatten())
            done_bool = float(done) if t < env._max_episode_steps else 0
            
            policy.rollout_buffer.rewards.insert(t,torch.tensor(reward).to(device).view(-1))
            
            state = next_state
            episode_reward += reward

            if done:
                print(f"Toral T:{step} Episode Num:{episode_nums} Reward:{episode_reward:.3f}")
                wandb.log({"train/reward":episode_reward}, step = step)
                last_step = t
                break

        with torch.no_grad():
            next_value = policy.get_value(np.array(state)).reshape(1,-1)
            advantages = torch.zeros(len(policy.rollout_buffer.rewards)).to(device)
            lastgaelam = 0
            for t in reversed(range(last_step + 1)):
                if t == last_step:
                    nextnonterminal = 1.0 - int(done)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - policy.rollout_buffer.is_terminals[t+1]
                    nextvalues = policy.rollout_buffer.state_values[t+1]
                delta = policy.rollout_buffer.rewards[t] + args.discount * nextvalues * nextnonterminal - policy.rollout_buffer.state_values[t]
                advantages[t] = lastgaelam = delta + args.discount * args.gae_lambda * nextnonterminal *  lastgaelam

            returns = advantages + torch.tensor(policy.rollout_buffer.state_values).to(device)
            #これはGAEのときにも通用する？GAEのときには実際に経験したリターンでは無くなってしまうがGAEによる予測がただしいとすればこれで良い．

        policy.train(args.batch_size, advantages, returns)

        policy.rollout_buffer.clear()


        if (episode_nums) % (args.eval_freq // args.max_ep_len)== 0:
            if (step+1) % args.record_freq == 0 and args.record_video:
                avg_reward = eval_policy(policy,args.env,args.seed,args.eval_episode,True)
            else:
                avg_reward = eval_policy(policy,args.env,args.seed,args.eval_episode,False)
            wandb.log({"eval/reward": avg_reward}, step=step)
            if (step+1) % args.save_freq == 0:
                policy.save(f"./models/{filename}")