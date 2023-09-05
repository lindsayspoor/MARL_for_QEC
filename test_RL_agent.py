import numpy as np
import os
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages ')
print(sys.path)
#import torch
import neat
#import gymnasium as gym

import config
import stable_baselines3
from toric_game_env import ToricGameEnv
from game import ToricCodeGame
from config import GameMode, RewardMode, ErrorModel
from perspectives import Perspectives
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


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


#game=ToricCodeGame(config.get_default_config())
env = ToricGameEnv(board_size, error_model, channels, config.get_default_config()['Training']['memory'])

model = PPO(MlpPolicy, env, verbose=0)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")



#q = env.state.qubit_pos
#done=False

#current_state = env.generate_errors(error_rate)


#print(f"current reward: {env.reward}")

#env.render()

#while not done:
#    redo_qubit = input("Give qubit number to redo the errors: ").split(',')
#    redo_qubit = [eval(i) for i in redo_qubit]
#    
#    img, reward, done, current_state = env.step(q[redo_qubit[0]], pauli_opt, without_illegal_actions=False)
    
#    print(f"current reward: {reward}")
#    print(current_state)
#    env.render()

