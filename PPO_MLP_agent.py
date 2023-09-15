import numpy as np
import matplotlib.pyplot as plt
import config
from stable_baselines3 import PPO, DQN
#from toric_game_env_new_errs import ToricGameEnv
from toric_game_env import ToricGameEnv
from config import GameMode, RewardMode, ErrorModel
#from perspectives import Perspectives
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import os
os.getcwd()






error_model = ErrorModel['UNCORRELATED']
max_steps = config.get_default_config()["Training"]["max_steps"]
epsilon = config.get_default_config()["Training"]["epsilon"]
rotation_invariant_decoder = config.get_default_config()["Training"]["rotation_invariant_decoder"]

if error_model == ErrorModel['UNCORRELATED']:
    channels=[0]
elif error_model == ErrorModel['DEPOLARIZING']:
    channels=[0,1]

'''
perspectives = Perspectives(board_size,
                    channels, config.get_default_config()['Training']['memory'])

'''

def evaluate_PPO_agent(model, number_evaluations, max_moves,logical_error_reward, success_reward,render):
    removed_syndromes=0
    moves_all=[]
    fail=0
    succes=0
    for k in range(number_evaluations):
        obs, info = env.reset()
        if render:
            env.render()
        moves=0
        for i in range(max_moves):
            action, _state = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action, without_illegal_actions=True)
            moves+=1
            #rewards+=reward
            if render:
                env.render()
            if done:
                if reward == logical_error_reward:
                    print("Game over, logical error!")
                    fail+=1
                    moves_all.append(moves)
                if reward==success_reward:
                    succes+=1
                    removed_syndromes+=1
                    if render:
                        env.render()
                    env.reset()
                    moves_all.append(moves)
                    moves=0
                break

        #print(f"successfully removed syndromes = {removed_syndromes}")
        #print(f"total number of moves = {moves}")
                #break

        #print(rewards)
        #print(info)
    mean_removed_syndromes = removed_syndromes/number_evaluations
    mean_moves = np.mean(moves_all)
    if ((succes+fail)==0):
        succes_rate=0
    else:
        succes_rate = succes/(succes+fail)

    return mean_removed_syndromes, mean_moves, succes_rate

def train_model(model, board_size,total_timesteps, learning_rate,error_rate,num_initial_errors, logical_error_reward, continue_reward, success_reward):
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    #model.save(f"trained_models/ppo_mlp_{total_timesteps}_timestep_board_{board_size}_error_rate_{error_rate}_lr_{learning_rate}_ler_{logical_error_reward}_cr_{continue_reward}_sr_{success_reward}")
    model.save(f"trained_models/ppo_mlp_{total_timesteps}_timestep_board_{board_size}_num_initial_errs_{num_initial_errors}_lr_{learning_rate}_ler_{logical_error_reward}_cr_{continue_reward}_sr_{success_reward}")
    return model

def initialise_model(model, board_size,total_timesteps, learning_rate, error_rate,num_initial_errors,logical_error_reward, continue_reward, success_reward):
    #model=PPO.load(f"trained_models/ppo_mlp_{total_timesteps}_timestep_board_{board_size}_error_rate_{error_rate}_lr_{learning_rate}_ler_{logical_error_reward}_cr_{continue_reward}_sr_{success_reward}")
    model=PPO.load(f"trained_models/ppo_mlp_{total_timesteps}_timestep_board_{board_size}_num_initial_errs_{num_initial_errors}_lr_{learning_rate}_ler_{logical_error_reward}_cr_{continue_reward}_sr_{success_reward}")
    return model


board_size = 5
error_rates=np.linspace(0.05, 0.25, 5)
error_rate=0.2
pauli_opt=0
num_initial_errors = 3
logical_error_reward=-1.0
success_reward=1.0
continue_reward=0.0
learning_rate=0.0005
total_timesteps=1000000
train=False
number_evaluations=1000
max_moves=200
render=False

'''
success_rates = []

for error_rate in error_rates:
    print(f"error rate = {error_rate}")

    env = ToricGameEnv(board_size, error_rate, num_initial_errors, logical_error_reward, continue_reward, success_reward,error_model, channels, config.get_default_config()['Training']['memory'])



    #EVALUATE PPO/DQN AGENT

    #evaluate un-trained agent
    model = PPO(MlpPolicy, env, learning_rate=learning_rate, verbose=0)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


    if train:
        model=train_model(model, board_size,total_timesteps, learning_rate, error_rate,num_initial_errors,logical_error_reward, continue_reward, success_reward)
    else:
        model=initialise_model(model, board_size,total_timesteps, learning_rate,error_rate, num_initial_errors, logical_error_reward, continue_reward, success_reward)


    mean_removed_syndromes, mean_moves, succes_rate = evaluate_PPO_agent(model, number_evaluations, max_moves, logical_error_reward, success_reward,render)

    success_rates.append(succes_rate)

    print(f"mean number of successfully removed syndromes = {mean_removed_syndromes}")
    print(f"mean total number of moves = {mean_moves}")
    print(f"error correction succes rate = {succes_rate} ")


    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"PPO N={num_initial_errors}")


success_rates=np.array(success_rates)

np.savetxt(f"MWPM/success_rate_ppo_mlp_timesteps_{total_timesteps}_lr_{learning_rate}_d_{board_size}_sr_{success_reward}_cr_{continue_reward}_ler_{logical_error_reward}.csv", success_rates,delimiter=',')

'''

env = ToricGameEnv(board_size, error_rate, num_initial_errors, logical_error_reward, continue_reward, success_reward,error_model, channels, config.get_default_config()['Training']['memory'])



#EVALUATE PPO/DQN AGENT

#evaluate un-trained agent
model = PPO(MlpPolicy, env, learning_rate=learning_rate, verbose=0)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if train:
    model=train_model(model, board_size,total_timesteps, learning_rate, error_rate,num_initial_errors,logical_error_reward, continue_reward, success_reward)
else:
    model=initialise_model(model, board_size,total_timesteps, learning_rate,error_rate, num_initial_errors, logical_error_reward, continue_reward, success_reward)


mean_removed_syndromes, mean_moves, succes_rate = evaluate_PPO_agent(model, number_evaluations, max_moves, logical_error_reward, success_reward,render)

#success_rates.append(succes_rate)

print(f"mean number of successfully removed syndromes = {mean_removed_syndromes}")
print(f"mean total number of moves = {mean_moves}")
print(f"error correction succes rate = {succes_rate} ")


# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print(f"PPO N={num_initial_errors}")

