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



board_size = config.get_default_config()["Physics"]["distance"]
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




env = ToricGameEnv(board_size, error_rate, num_initial_errors, error_model, channels, config.get_default_config()['Training']['memory'])



#evaluate un-trained agent
model = PPO(MlpPolicy, env, verbose=0)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)



print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
total_timesteps=5000_000
model.learn(total_timesteps=total_timesteps, progress_bar=True)
model.save(f"ppo_mlp_{total_timesteps}_timesteps")
#model=PPO.load(f"ppo_mlp_{total_timesteps}_timesteps")

obs, info = env.reset()
env.render()
rewards=0
for i in range(100):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action, without_illegal_actions=True)
    #rewards+=reward
    env.render()
    if done:
        if reward == -1:
            print("Game over, logical error!")
            break
        rewards+=reward
        env.reset()
        env.render()
print(f"total score = {rewards}")
        #break

print(rewards)
print(info)

#plt.plot(range(1000), rewards, label='reward per episode')
#plt.xlabel("episode")
#plt.ylabel("reward")
#plt.legend()
#plt.show()


# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")




