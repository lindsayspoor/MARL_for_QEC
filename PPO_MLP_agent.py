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

def plot_illegal_action_rate(error_rates, illegal_action_rates, path, error_rates_curriculum):
    plt.figure()
    for j in range(illegal_action_rates.shape[0]):
        plt.scatter(error_rates, illegal_action_rates[j,:], label=f'p_error={error_rates_curriculum[j]}')
        plt.plot(error_rates, illegal_action_rates[j,:], linestyle='-.', linewidth=0.5)
    plt.title(r'Toric Code - Illegal action rate')
    plt.xlabel(r'$p_x$')
    plt.ylabel(r'Illegal actions[\%]')
    plt.legend()
    plt.savefig(f'Figure_results/Results_illegal_actions/benchmark_MWPM_curriculum_{path}.pdf')
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
        #self.model = PPO(MlpPolicy, self.env, learning_rate=self.initialisation_settings['learning_rate'], verbose=0)
        self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, learning_rate=self.initialisation_settings['learning_rate'], verbose=0)
        print("initialisation done")

    def change_environment_settings(self, settings):
        print("changing environment settings...")
        if settings['random_error_distribution']:
            self.env = ToricGameEnv(settings)
        else: 
            self.env = ToricGameEnvLocalErrs(settings)
        self.model.set_env(self.env)
        print("changing settings done")

    def train_model(self, save_model_path):
        print("training the model...")
        self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True)
        self.model.save(f"trained_models/ppo_{save_model_path}")
        print("training done")

    def load_model(self, load_model_path):
        print("loading the model...")
        #self.model=PPO.load(f"trained_models/ppo_{load_model_path}")
        self.model=MaskablePPO.load(f"trained_models/ppo_{load_model_path}")
        print("loading done")

    def evaluate_model(self, evaluation_settings, render, number_evaluations, max_moves):
        print("evaluating the model...")
        moves=0
        logical_errors=0
        success=0
        maximum=0
        #illegal=0
        for k in range(number_evaluations):
            obs, info = self.env.reset()
            if render:
                self.env.render()
            for i in range(max_moves):
                action_masks=get_action_masks(self.env)
                action, _state = self.model.predict(obs, action_masks=action_masks)
                obs, reward, done, truncated, info = self.env.step(action, without_illegal_actions=True)
                moves+=1
                if render:
                    self.env.render()
                if done:
                    if reward == evaluation_settings['logical_error_reward']:
                        #print(info['message'])
                        logical_errors+=1
                    #if reward == evaluation_settings['illegal_action_reward']:
                        #print(info['message'])
                        #illegal+=1
                    if reward == evaluation_settings['success_reward']:
                        success+=1
                        self.env.reset()
                    break
                

                    
            
        print(f"mean number of moves per evaluation is {moves/number_evaluations}")
        
        if (success+logical_errors)==0:
            success_rate = 0
        else:
            success_rate= success / (success+logical_errors)

        #if (success+logical_errors+illegal)==0:
            #illegal_action_rate = 0
        #else:
            #illegal_action_rate = (illegal/(success+logical_errors+illegal))

        print("evaluation done")

        return success_rate#, illegal_action_rate

#SETTINGS FOR RUNNING THIS SCRIPT
train=True
curriculum=False
benchmark_MWPM=True
save_files=True
render=False
number_evaluations=1000
max_moves=200
evaluate=True

board_size=5
error_rate=0.1
logical_error_reward=-1000
success_reward=1000
continue_reward=0.0
illegal_action_reward=-1000
total_timesteps=6e6
random_error_distribution=False
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



success_rates_all=[]
#illegal_action_rates_all=[]
#error_rates_curriculum=np.arange(0.1, 0.2, 0.05)[1:]
#error_rates_curriculum=[0.1]
error_rate_curriculum=0.1
#for error_rate_curriculum in error_rates_curriculum:
#logical_error_rewards=[-500, -1000, -1500, -2000]
#logical_error_rewards=[-1000]
#success_rewards=[500, 800, 1000, 2000]
success_rewards=[1000]
for success_reward in success_rewards:
#for logical_error_reward in logical_error_rewards:
    print(f"{success_reward=}")
    #print(f"{logical_error_reward=}")

    #initialisation_settings['logical_error_reward']=logical_error_reward
    #loaded_model_settings['logical_error_reward']=logical_error_reward
    initialisation_settings['success_reward']=success_reward
    loaded_model_settings['success_reward']=success_reward

    save_model_path =''
    for key, value in initialisation_settings.items():
        save_model_path+=f"{key}={value}"

    #loaded_model_settings['random_error_distribution']=True
    load_model_path =''
    for key, value in loaded_model_settings.items():
        load_model_path+=f"{key}={value}"

    #loaded_model_settings['random_error_distribution']=False

    #initialise PPO Agent
    AgentPPO = PPO_agent(initialisation_settings)

    if train:
        AgentPPO.train_model(save_model_path=save_model_path)
    else:
        print(f"{loaded_model_settings['error_rate']=}")
        AgentPPO.load_model(load_model_path=load_model_path)

    if curriculum:
        print(f"{error_rate_curriculum=}")
        initialisation_settings['error_rate']=error_rate_curriculum
        save_model_path =''
        for key, value in initialisation_settings.items():
            save_model_path+=f"{key}={value}"

        AgentPPO.change_environment_settings(initialisation_settings)

        AgentPPO.train_model(save_model_path=save_model_path)
        loaded_model_settings['error_rate']=error_rate_curriculum

            

    p_start = 0.01 
    p_end = 0.20
    error_rates = np.linspace(p_start,p_end,6)
    #error_rates=[0.1]


    if evaluate:
        success_rates=[]
        illegal_action_rates=[]
        for error_rate in error_rates:
            #SET SETTINGS TO EVALUATE LOADED AGENT ON
            print(f"{error_rate=}")
            evaluation_settings['error_rate'] = error_rate

            AgentPPO.change_environment_settings(evaluation_settings)
            success_rate = AgentPPO.evaluate_model(evaluation_settings, render, number_evaluations, max_moves)
            success_rates.append(success_rate)
            #illegal_action_rates.append(illegal_action_rate)
            print(f"{success_rate=}")
            #print(f"{illegal_action_rate=}")


        success_rates=np.array(success_rates)
        #illegal_action_rates=np.array(illegal_action_rates)

        evaluation_path =''
        for key, value in evaluation_settings.items():
            evaluation_path+=f"{key}={value}"

        if save_files:
            if curriculum:
                np.savetxt(f"Files_results/files_success_rates/success_rates_ppo_{evaluation_path}_curr={error_rate_curriculum:.3f}.csv", success_rates)
    success_rates_all.append(success_rates)
    #illegal_action_rates_all.append(illegal_action_rates)


simulation_settings = {'decoder': 'MWPM',
                    'N': 1000,
                    'delta_p': 0.001,
                    'p_start': p_start,
                    'p_end': p_end,
                    'path': f'Figure_results/Results_benchmarks/benchmark_MWPM_curriculum_{evaluation_path}.pdf',
                    'tex_plot' : False,
                    'save_data' : True,
                    'plot_all' : True,
                    'all_L':[board_size],
                    'random_errors':random_error_distribution,
                    'lambda_value':lambda_value}

plot_settings = simulation_settings
plot_settings['all_L']=[board_size]

success_rates_all=np.array(success_rates_all)
#illegal_action_rates_all=np.array(illegal_action_rates_all)



#success_rates_005 = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=-1000success_reward=1000continue_reward=0.0illegal_action_reward=-800learning_rate=0.0005total_timesteps=3000000.0random_error_distribution=Trueaction_mask=True_curr=0.05.csv")
#success_rates_all=np.vstack((success_rates_all, success_rates_005))

#success_rates_01 = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=-1000success_reward=1000continue_reward=0.0illegal_action_reward=-800learning_rate=0.0005total_timesteps=3000000.0random_error_distribution=Trueaction_mask=True_curr=0.1.csv")
#success_rates_all=np.vstack((success_rates_all, success_rates_01))

#success_rewards.append(1000)
#logical_error_rewards.append(-1000)
#error_rates_curriculum.append(0.05)
#error_rates_curriculum.append(0.1)
if benchmark_MWPM:
    sim_data, sim_all_data = simulate(simulation_settings)
    #plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, logical_error_rewards)
    #plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, error_rates_curriculum)
    plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, success_rewards)

#if plot_illegal_actions_rate:
#    plot_illegal_action_rate(error_rates, illegal_action_rates_all, evaluation_path, error_rates_curriculum)
