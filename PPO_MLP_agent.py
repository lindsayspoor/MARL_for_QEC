import numpy as np
import os
import sys
#import torch
#import neat
import gym



import matplotlib.pyplot as plt
import config
from stable_baselines3 import PPO
from toric_game_env import ToricGameEnv
from game import ToricCodeGame
from config import GameMode, RewardMode, ErrorModel
from perspectives import Perspectives
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat



#board_size = config.get_default_config()["Physics"]["distance"]
board_size = 3
#error_model = config.get_default_config()["Physics"]["error_model"]
error_model = ErrorModel['UNCORRELATED']
max_steps = config.get_default_config()["Training"]["max_steps"]
epsilon = config.get_default_config()["Training"]["epsilon"]
rotation_invariant_decoder = config.get_default_config()["Training"]["rotation_invariant_decoder"]

if error_model == ErrorModel['UNCORRELATED']:
    channels=[0]
elif error_model == ErrorModel['DEPOLARIZING']:
    channels=[0,1]

perspectives = Perspectives(board_size,
                    channels, config.get_default_config()['Training']['memory'])

error_rate=0.2
pauli_opt=0
num_initial_errors = 3

def evaluate_PPO_agent(model, number_evaluations, render):
    removed_syndromes=0
    moves=0
    for k in range(number_evaluations):
        obs, info = env.reset()
        if render:
            env.render()

        for i in range(10):
            action, _state = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action, without_illegal_actions=True)
            moves+=1
            #rewards+=reward
            if render:
                env.render()
            if done:
                if reward == -1:
                    print("Game over, logical error!")
                    break
                removed_syndromes+=1
                if reward==1:
                    if render:
                        env.render()
                    env.reset()

        #print(f"successfully removed syndromes = {removed_syndromes}")
        #print(f"total number of moves = {moves}")
                #break

        #print(rewards)
        #print(info)
    mean_removed_syndromes = removed_syndromes/number_evaluations
    mean_moves = moves/number_evaluations

    return mean_removed_syndromes, mean_moves


env = ToricGameEnv(board_size, error_rate, num_initial_errors, error_model, channels, config.get_default_config()['Training']['memory'])



#EVALUATE PPO AGENT

#evaluate un-trained agent
learning_rate=0.0005

model = PPO(MlpPolicy, env, learning_rate=learning_rate, verbose=0)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)



print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for ... steps
total_timesteps=1000000
model.learn(total_timesteps=total_timesteps, progress_bar=True)
model.save(f"ppo_mlp_{total_timesteps}_timestep_lr_{learning_rate}_incentive_punishment")
#model=PPO.load(f"ppo_mlp_{total_timesteps}_timestep_lr_{learning_rate}")

number_evaluations=100

mean_removed_syndromes, mean_moves = evaluate_PPO_agent(model, number_evaluations, render=False)


print(f"mean number of successfully removed syndromes = {mean_removed_syndromes}")
print(f"mean total number of moves = {mean_moves}")


# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


'''
plt.figure()
plt.plot(np.linspace(0.00005, 0.001, 10), mean_removed_syndromes_lr)
plt.xlabel("learning rate")
plt.ylabel("mean removed syndromes")
plt.show()

plt.figure()
plt.plot(np.linspace(0.00005, 0.001, 10), mean_moves_lr)
plt.xlabel("learning rate")
plt.ylabel("mean number of moves")
plt.show()
'''


#EVALUATE MWPM AGENT


