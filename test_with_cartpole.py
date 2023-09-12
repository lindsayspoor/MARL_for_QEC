import numpy as np
import os
import sys
#import torch
#import neat
import gymnasium as gym



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
            #if render:
                #env.render()
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


env = gym.make("CartPole-v1")

model = PPO(MlpPolicy, env, verbose=0)



mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)



print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
# Train the agent for 10000 steps
model.learn(total_timesteps=10_000)


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


