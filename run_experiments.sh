#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1))
do 
	python algo/TD3/main_TD3.py \
	--policy "TD3" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--expl_noise 0.1 \
	--policy_noise 0.2 \
	--delay 2 \
	--noise_clip 0.5 

	python algo/DDPG/main_DDPG.py \
	--policy "DDPG" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--expl_noise 0.1 \
	--policy_noise 0.2 \
	--delay 2 \
	--noise_clip 0.5

	python algo/SAC/main_SAC.py \
	--policy "SAC" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--alpha_tuning \

	python algo/PPO/main_PPO.py \
	--policy "PPO" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--max_ep_len 1000 \
	--update_epochs 4 \
	--clip_coef 0.2 \
	--target_kl 0.1 \
	--norm_adv \
	--ent_coef 0.0 \
	--max_grad_norm 0.5 \
	--anneal_lr \
	--gae_lambda 0.9

	python algo/QTOpt/main_QT_Opt.py \
	--policy "QT_Opt" \
	--env "HalfCheetah-v2" \
	--seed $i \
	--epsilon 0.2 \
	--cem_iter 2 \
	--select_num 5 \
	--num_samples 64

done
