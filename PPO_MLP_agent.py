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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

class PPO_agent:
    def __init__(self, initialisation_settings, log):#path):

        self.initialisation_settings=initialisation_settings
        # Create log dir
        self.log=log
        if self.log:
            self.log_dir = "log_dir"
            os.makedirs(self.log_dir, exist_ok=True)



        #INITIALISE MODEL FOR INITIALISATION
        self.initialise_model()

    def initialise_model(self):
        #INITIALISE ENVIRONMENT INITIALISATION
        print("initialising the environment and model...")
        if self.initialisation_settings['random_error_distribution']:
            if self.initialisation_settings['fixed']:
                self.env = ToricGameEnvFixedErrs(self.initialisation_settings)
            else:
                self.env = ToricGameEnv(self.initialisation_settings)
        else: 
            self.env = ToricGameEnvLocalErrs(self.initialisation_settings)

        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every 1000 steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
            
        #INITIALISE MODEL FOR INITIALISATION
        #self.model = PPO(MlpPolicy, self.env, learning_rate=self.initialisation_settings['learning_rate'], verbose=0)
        self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, learning_rate=self.initialisation_settings['learning_rate'], verbose=0)
        print("initialisation done")

    def change_environment_settings(self, settings):
        print("changing environment settings...")
        if settings['random_error_distribution']:
            if settings['fixed']:
                self.env = ToricGameEnvFixedErrs(settings)
            else:
                self.env = ToricGameEnv(settings)
        else: 
            self.env = ToricGameEnvLocalErrs(settings)
        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every 1000 steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)
        
        self.model.set_env(self.env)

        print("changing settings done")

    def train_model(self, save_model_path):
        print("training the model...")
        if self.log:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True, callback=self.callback)
            self.plot_results(self.log_dir, save_model_path)
        else:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True)
    
        self.model.save(f"trained_models/ppo_{save_model_path}")
        print("training done")

    def load_model(self, load_model_path):
        print("loading the model...")
        #self.model=PPO.load(f"trained_models/ppo_{load_model_path}")

        self.model=MaskablePPO.load(f"trained_models/ppo_{load_model_path}")
        print("loading done")
    
    def moving_average(self,values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        print(values)
        return np.convolve(values, weights, "valid")


    def plot_results(self,log_folder, save_model_path, title="Learning Curve"):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = self.moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y) :]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.title(title + " Smoothed")
        plt.savefig(f'Figure_results/Results_reward_logs/learning_curve_{save_model_path}.pdf')
        plt.show()

    def evaluate_model(self, evaluation_settings, render, number_evaluations, max_moves, check_fails):
        print("evaluating the model...")
        moves=0
        logical_errors=0
        success=0
        maximum=0
        #illegal=0
        for k in range(number_evaluations):
            obs, info = self.env.reset()
            obs0=obs
            #self.env.render()
            if render:
                self.env.render()
            for i in range(max_moves):
                action_masks=get_action_masks(self.env)
                #print(f"{action_masks=}")
                action, _state = self.model.predict(obs, action_masks=action_masks)
                #print(f"{action=}")
                obs, reward, done, truncated, info = self.env.step(action)#, without_illegal_actions=True)
                moves+=1
                if render:
                    self.env.render()
                if done:
                    if reward == evaluation_settings['logical_error_reward']:
                        if check_fails:
                            print(info['message'])
                            print(obs0)
                            #self.env.render()
                        logical_errors+=1
                        
                    #if reward == evaluation_settings['illegal_action_reward']:
                        #print(info['message'])
                        #illegal+=1
                    if reward == evaluation_settings['success_reward']:
                        success+=1
                        #self.env.render()
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


    def evaluate_fixed_errors(self, evaluation_settings, N_evaluates, render, number_evaluations, max_moves, check_fails, save_files):
        
        success_rates=[]

        for N_evaluate in N_evaluates:
            print(f"{N_evaluate=}")
            evaluation_settings['fixed'] = evaluate_fixed
            evaluation_settings['N']=N_evaluate
            self.change_environment_settings(evaluation_settings)
            success_rate = self.evaluate_model(evaluation_settings, render, number_evaluations, max_moves, check_fails)
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
            np.savetxt(f"Files_results/files_success_rates/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates)

        return success_rates
    

    def evaluate_error_rates(self,evaluation_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files):
        success_rates=[]

        for error_rate in error_rates:
            #SET SETTINGS TO EVALUATE LOADED AGENT ON
            print(f"{error_rate=}")
            evaluation_settings['error_rate'] = error_rate
            evaluation_settings['fixed'] = evaluate_fixed

            self.change_environment_settings(evaluation_settings)
            success_rate = self.evaluate_model(evaluation_settings, render, number_evaluations, max_moves, check_fails)
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
            np.savetxt(f"Files_results/files_success_rates/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates)

        return success_rates



#SETTINGS FOR RUNNING THIS SCRIPT
train=False
curriculum=False
benchmark_MWPM=True
save_files=True
render=False
number_evaluations=10000
max_moves=200
evaluate=True
check_fails=False

board_size=5
error_rate=0.01
logical_error_reward=5
success_reward=10
continue_reward=-1
illegal_action_reward=-1000
total_timesteps=600000
random_error_distribution=True
mask_actions=True
log = True
lambda_value=1
fixed=True
evaluate_fixed=False
N_evaluate=4
N=4


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
            'lambda_value': lambda_value,
            'fixed':fixed,
            'N':N
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
            'lambda_value': lambda_value,
            'fixed':fixed,
            'N':N
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
            'lambda_value': lambda_value,
            'fixed':fixed,
            'N':N
            }



success_rates_all=[]

error_rates_curriculum=[0.01]
N_curriculums=[4]

#for error_rate_curriculum in error_rates_curriculum:
for N_curriculum in N_curriculums:

    #initialisation_settings['error_rate']=error_rate_curriculum
    #loaded_model_settings['error_rate']=error_rate_curriculum
    #initialisation_settings['N']=N_curriculum
    #loaded_model_settings['N']=N_curriculum

    save_model_path =''
    for key, value in initialisation_settings.items():
        save_model_path+=f"{key}={value}"

    #loaded_model_settings['random_error_distribution']=True
    load_model_path =''
    for key, value in loaded_model_settings.items():
        load_model_path+=f"{key}={value}"

    #loaded_model_settings['random_error_distribution']=False


    #initialise PPO Agent
    AgentPPO = PPO_agent(initialisation_settings, log)

    if train:
        AgentPPO.train_model(save_model_path=save_model_path)
    else:
        print(f"{loaded_model_settings['error_rate']=}")
        AgentPPO.load_model(load_model_path=load_model_path)

    if curriculum:
        #print(f"{error_rate_curriculum=}")
        print(f"{N_curriculum=}")
        #initialisation_settings['error_rate']=error_rate_curriculum
        initialisation_settings['N']=N_curriculum
        initialisation_settings['total_timesteps']=600000
        save_model_path =''
        for key, value in initialisation_settings.items():
            save_model_path+=f"{key}={value}"

        AgentPPO.change_environment_settings(initialisation_settings)

        AgentPPO.train_model(save_model_path=save_model_path)
        #loaded_model_settings['error_rate']=error_rate_curriculum
        loaded_model_settings['N']=N_curriculum

            

    p_start = 0.01 
    p_end = 0.20
    error_rates = np.linspace(p_start,p_end,6)
    N_evaluates = [4]
    #error_rates=[0.05]


    if evaluate:

        if evaluate_fixed:
            success_rates = AgentPPO.evaluate_fixed_errors(evaluation_settings, N_evaluates, render, number_evaluations, max_moves, check_fails, save_files)
        else:
            success_rates = AgentPPO.evaluate_error_rates(evaluation_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files)


        success_rates_all.append(success_rates)


evaluation_path =''
for key, value in evaluation_settings.items():
    evaluation_path+=f"{key}={value}"

if fixed:
    path = f"Figure_results/Results_benchmarks/benchmark_MWPM_{evaluation_path}_{loaded_model_settings['N']}.pdf"
else:
    path = f"Figure_results/Results_benchmarks/benchmark_MWPM_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"

simulation_settings = {'decoder': 'MWPM',
                    'N': 1000,
                    'delta_p': 0.001,
                    'p_start': p_start,
                    'p_end': p_end,
                    'path': path,
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



#success_rates_001 = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=3000000.0random_error_distribution=Truemask_actions=Truelambda_value=1_0.01.csv")
#success_rates_all=np.vstack((success_rates_all, success_rates_001))

#success_rates_005 = np.loadtxt(f"Files_results/files_success_rates/success_rates_ppo_board_size=5error_model=0error_rate=0.2logical_error_reward=5success_reward=10continue_reward=-1learning_rate=0.0005total_timesteps=3000000.0random_error_distribution=Truemask_actions=Truelambda_value=1_0.05.csv")
#success_rates_all=np.vstack((success_rates_all, success_rates_005))


#error_rates_curriculum.append(0.01)
#error_rates_curriculum.append(0.05)

if benchmark_MWPM:
    sim_data, sim_all_data = simulate(simulation_settings)
    plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, error_rates_curriculum)

#if plot_illegal_actions_rate:
#    plot_illegal_action_rate(error_rates, illegal_action_rates_all, evaluation_path, error_rates_curriculum)
