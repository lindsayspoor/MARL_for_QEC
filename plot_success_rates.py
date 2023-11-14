import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from toric_game_env import ToricGameEnv, ToricGameEnvLocalErrs, ToricGameEnvFixedErrs, ToricGameEnvLocalFixedErrs
from config import ErrorModel
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
#from stable_baselines3.common.evaluation import evaluate_policy
from simulate_MWPM import simulate, plot
import os
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
os.getcwd()



#SETTINGS FOR RUNNING THIS SCRIPT
train=False
curriculum=False
benchmark_MWPM=True
save_files=True
render=False
number_evaluations=10000
max_moves=200
evaluate=True

board_size=5
error_rate=0.05
logical_error_reward=5
success_reward=10
continue_reward=-1
illegal_action_reward=-1000
total_timesteps=3e6
random_error_distribution=True
mask_actions=True
lambda_value=1


#SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'logical_error_reward': logical_error_reward,
            'success_reward': success_reward,
            'continue_reward':continue_reward,
            #'illegal_action_reward':illegal_action_reward,
            'learning_rate':0.0005,
            'total_timesteps': total_timesteps,
            'random_error_distribution': random_error_distribution,
            'mask_actions': mask_actions,
            'lambda_value': lambda_value
            }

#SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'logical_error_reward': logical_error_reward,
            'success_reward': success_reward,
            'continue_reward':continue_reward,
            #'illegal_action_reward':illegal_action_reward,
            'learning_rate':0.0005,
            'total_timesteps': total_timesteps,
            'random_error_distribution': random_error_distribution,
            'mask_actions': mask_actions,
            'lambda_value': lambda_value
            }

evaluation_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'logical_error_reward': logical_error_reward,
            'success_reward': success_reward,
            'continue_reward':continue_reward,
            #'illegal_action_reward':illegal_action_reward,
            'learning_rate':0.0005,
            'total_timesteps': total_timesteps,
            'random_error_distribution': random_error_distribution,
            'mask_actions': mask_actions,
            'lambda_value': lambda_value
            }

evaluation_path =''
for key, value in evaluation_settings.items():
    evaluation_path+=f"{key}={value}"

p_start = 0.01 
p_end = 0.20
error_rates = np.linspace(p_start,p_end,6)
#error_rates=[0.05]
simulation_settings = {'decoder': 'MWPM',
                    'N': 1000,
                    'delta_p': 0.001,
                    'p_start': p_start,
                    'p_end': p_end,
                    'path': f'Figure_results/Results_benchmarks/benchmark_MWPM_success_rates_error_rates.pdf',
                    'tex_plot' : False,
                    'save_data' : True,
                    'plot_all' : True,
                    'all_L':[board_size],
                    'random_errors':random_error_distribution,
                    'lambda_value':lambda_value}


plot_settings = simulation_settings
plot_settings['all_L']=[board_size]


success_rates_all = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=300000random_error_distribution=Truemask_actions=Truelambda_value=1fixed=FalseN=1_0.01.csv")
#success_rates_all=np.vstack((success_rates_all, success_rates_all))

success_rates = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=300000random_error_distribution=Truemask_actions=Truelambda_value=1fixed=FalseN=1_0.048.csv")
success_rates_all=np.vstack((success_rates_all, success_rates))
success_rates = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=300000random_error_distribution=Truemask_actions=Truelambda_value=1fixed=FalseN=1_0.086.csv")
success_rates_all=np.vstack((success_rates_all, success_rates))
success_rates = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=300000random_error_distribution=Truemask_actions=Truelambda_value=1fixed=FalseN=1_0.12399999999999999.csv")
success_rates_all=np.vstack((success_rates_all, success_rates))
success_rates = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=300000random_error_distribution=Truemask_actions=Truelambda_value=1fixed=FalseN=1_0.162.csv")
success_rates_all=np.vstack((success_rates_all, success_rates))
success_rates = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=300000random_error_distribution=Truemask_actions=Truelambda_value=1fixed=FalseN=1_0.2.csv")
success_rates_all=np.vstack((success_rates_all, success_rates))
error_rates_curriculum=np.linspace(0.01,0.20,6)

#reward scheme = 0 = new reward settings, reward scheme = 1 = old settings
if benchmark_MWPM:
    sim_data, sim_all_data = simulate(simulation_settings)
    #plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, logical_error_rewards)
    plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, error_rates_curriculum)
    #plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, continue_rewards)