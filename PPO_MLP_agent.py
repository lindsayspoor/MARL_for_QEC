import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from toric_game_env import ToricGameEnv, ToricGameEnvLocalErrs, ToricGameEnvFixedErrs, ToricGameEnvLocalFixedErrs
from config import ErrorModel
from stable_baselines3.ppo.policies import MlpPolicy
#from stable_baselines3.common.evaluation import evaluate_policy
from simulate_MWPM import simulate, plot
import os
import torch as th
os.getcwd()

def plot_illegal_action_rate(error_rates, illegal_action_rates, path):
    plt.figure()
    plt.scatter(error_rates, illegal_action_rates)
    plt.plot(error_rates, illegal_action_rates, linestyle='-.', linewidth=0.5)
    plt.title(r'Toric Code - Illegal action rate')
    plt.xlabel(r'$p_x$')
    plt.ylabel(r'Illegal actions[\%]')
    plt.savefig(f'Figure_results/Results_benchmarks/benchmark_MWPM_{path}.pdf')
    plt.show()


class PPO_agent:
    def __init__(self, initialisation_settings):#path):

        self.initialisation_settings=initialisation_settings

        #INITIALISE MODEL FOR INITIALISATION
        self.initialise_model()

    def initialise_model(self):
        #INITIALISE ENVIRONMENT INITIALISATION
        print("initialising the environment and model...")
        if self.initialisation_settings['random_error_distribution']:
            self.env = ToricGameEnv(self.initialisation_settings)
        else: 
            self.env = ToricGameEnvLocalErrs(self.initialisation_settings)
            
        #INITIALISE MODEL FOR INITIALISATION
        self.model = PPO(MlpPolicy, self.env, learning_rate=self.initialisation_settings['learning_rate'], verbose=0)
        print("initialisation done")

    def change_environment_settings(self, settings):
        print("changing environment settings...")
        if settings['random_error_distribution']:
            self.env = ToricGameEnv(settings)
        else: 
            self.env = ToricGameEnvLocalErrs(settings)
        print("changing settings done")

    def train_model(self, save_model_path):
        print("training the model...")
        self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True)
        self.model.save(f"trained_models/ppo_{save_model_path}")
        print("training done")

    def load_model(self, load_model_path):
        print("loading the model...")
        self.model=PPO.load(f"trained_models/ppo_{load_model_path}")
        print("loading done")

    def evaluate_model(self, evaluation_settings, render, number_evaluations, max_moves):
        print("evaluating the model...")
        moves=0
        logical_errors=0
        success=0
        illegal=0
        for k in range(number_evaluations):
            obs, info = self.env.reset()
            if render:
                self.env.render()
            for i in range(max_moves):
                action, _state = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action, without_illegal_actions=True)
                moves+=1
                if render:
                    self.env.render()
                if done:
                    if reward == evaluation_settings['logical_error_reward']:
                        print(info['message'])
                        logical_errors+=1
                    if reward == evaluation_settings['illegal_action_reward']:
                        print(info['message'])
                        illegal+=1
                    if reward == evaluation_settings['success_reward']:
                        success+=1
                        self.env.reset()
                    break
            
        print(f"mean number of moves per evaluation is {moves/number_evaluations}")
        
        if (success+logical_errors)==0:
            success_rate = 0
        else:
            success_rate= success / (success+logical_errors)

        illegal_action_rate = (illegal/(success+logical_errors+illegal))

        print("evaluation done")

        return success_rate, illegal_action_rate

#SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': 5,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': 0.1,
            'logical_error_reward': -1000,
            'success_reward': 1000,
            'continue_reward':0.0,
            'illegal_action_reward':-800,
            'learning_rate':0.0005,
            'total_timesteps': 3e6,
            #'with_error_rates': True,
            'random_error_distribution': True,
            'action_mask': True,
            }

#SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': 5,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': 0.1,
            'logical_error_reward': -1000,
            'success_reward': 1000,
            'continue_reward':0.0,
            'illegal_action_reward':-800,
            'learning_rate':0.0005,
            'total_timesteps': 3e6,
            #'with_error_rates': True,
            'random_error_distribution': True,
            'action_mask': True,
            }

evaluation_settings = {'board_size': 5,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': 0.1,
            'logical_error_reward': -1000,
            'success_reward': 1000,
            'continue_reward':0.0,
            'illegal_action_reward':-800,
            'learning_rate':0.0005,
            'total_timesteps': 3e6,
            #'with_error_rates': True,
            'random_error_distribution': True,
            'action_mask': True,
            }

#SETTINGS FOR RUNNING THIS SCRIPT
train=False
curriculum=False
benchmark_MWPM=True
plot_illegal_actions_rate = True
save_files=True
render=False
number_evaluations=1000
max_moves=200
evaluate=True

save_model_path =''
for key, value in initialisation_settings.items():
    save_model_path+=f"{key}={value}"

load_model_path =''
for key, value in loaded_model_settings.items():
    load_model_path+=f"{key}={value}"

#initialise PPO Agent
AgentPPO = PPO_agent(initialisation_settings)

if train:
    AgentPPO.train_model(save_model_path=save_model_path)
else:
    AgentPPO.load_model(load_model_path=load_model_path)

p_start = 0.01 
p_end = 0.20
error_rates = np.linspace(p_start,p_end,6)

#error_rates=[0.1]

#setting the action mask constraint to False when evaluating so that it is allowed to take any action the agent likes
evaluation_settings['action_mask']=False

if evaluate:
    success_rates=[]
    illegal_action_rates=[]
    for error_rate in error_rates:
        #SET SETTINGS TO EVALUATE LOADED AGENT ON
        print(f"{error_rate=}")
        evaluation_settings['error_rate'] = error_rate

        AgentPPO.change_environment_settings(evaluation_settings)
        success_rate, illegal_action_rate = AgentPPO.evaluate_model(evaluation_settings, render, number_evaluations, max_moves)
        success_rates.append(success_rate)
        illegal_action_rates.append(illegal_action_rate)
        print(f"{success_rate=}")
        print(f"{illegal_action_rate=}")


    success_rates=np.array(success_rates)
    illegal_action_rates=np.array(illegal_action_rates)

    evaluation_path=load_model_path

    if save_files:
        np.savetxt(f"Files_results/files_success_rates/success_rates_ppo_{evaluation_path}.csv", success_rates)

simulation_settings = {'decoder': 'MWPM',
                       'N': 1000,
                       'delta_p': 0.001,
                       'p_start': p_start,
                       'p_end': p_end,
                       'path': f'Figure_results/Results_benchmarks/benchmark_MWPM_{evaluation_path}.pdf',
                       'tex_plot' : False,
                       'save_data' : True,
                       'plot_all' : True,
                       'all_L':[5]}

plot_settings = simulation_settings
plot_settings['all_L']=[5]

if benchmark_MWPM:
    sim_data, sim_all_data = simulate(simulation_settings)
    plot(plot_settings, sim_data, sim_all_data, success_rates, error_rates)

if plot_illegal_actions_rate:
    plot_illegal_action_rate(error_rates, illegal_action_rates, evaluation_path)
