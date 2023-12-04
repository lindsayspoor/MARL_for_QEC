import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from toric_game_dynamic_env import ToricGameDynamicEnv,  ToricGameDynamicEnvFixedErrs
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
import networkx as nx
import pandas as pd 
os.getcwd()

def plot_illegal_action_rate(N_evaluates, illegal_action_rates, path, error_rates_curriculum):
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


def plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM,error_rates_eval, board_size,path_plot,agent_value_N, agent_value_error_rate,evaluate_fixed):
    plt.figure()
    #for j in range(success_rates.shape[0]):
    if evaluate_fixed:
        plt.plot(N_evaluates, success_rates_all_MWPM[0,:]*100, label=f'd={board_size} MWPM decoder', linestyle='-.', linewidth=0.5, color='black')
        plt.scatter(N_evaluates, success_rates_all[0,:]*100, label=f"d={board_size} PPO agent, N={agent_value_N}", marker="^", s=30)
        plt.plot(N_evaluates, success_rates_all[0,:]*100, linestyle='-.', linewidth=0.5)
    else:
        plt.plot(error_rates_eval, success_rates_all_MWPM[0,:]*100, label=f'd={board_size} MWPM decoder', linestyle='-.', linewidth=0.5, color='black')
        plt.scatter(error_rates_eval, success_rates_all[0,:]*100, label=f"d={board_size} PPO agent, p_error={agent_value_error_rate}", marker="^", s=30)
        plt.plot(error_rates_eval, success_rates_all[0,:]*100, linestyle='-.', linewidth=0.5)
    plt.title(r'Toric Code - PPO vs MWPM')
    plt.xlabel(r'N')
    plt.ylabel(r'Correct[\%] $p_s$')
    plt.legend()
    plt.savefig(path_plot)
    #plt.show()

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
            self.log_dir = "log_dir_dynamic"
            os.makedirs(self.log_dir, exist_ok=True)



        #INITIALISE MODEL FOR INITIALISATION
        self.initialise_model()

    def initialise_model(self):
        #INITIALISE ENVIRONMENT INITIALISATION
        print("initialising the environment and model...")
        if self.initialisation_settings['fixed']:
            self.env = ToricGameDynamicEnvFixedErrs(self.initialisation_settings)
        else:
            self.env = ToricGameDynamicEnv(self.initialisation_settings)





        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every 1000 steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
            
        #INITIALISE MODEL FOR INITIALISATION
        if self.initialisation_settings['mask_actions']:
            self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, learning_rate=self.initialisation_settings['lr'], verbose=0, policy_kwargs={"net_arch":dict(pi=[64, 64], vf=[64, 64])})
        else:
            self.model = PPO(MlpPolicy, self.env, learning_rate=self.initialisation_settings['lr'], verbose=0, policy_kwargs={"net_arch":dict(pi=[64, 64], vf=[64, 64])})
        print("initialisation done")
        print(self.model.policy)

    def change_environment_settings(self, settings):
        print("changing environment settings...")

        if settings['fixed']:
            self.env = ToricGameDynamicEnvFixedErrs(settings)
        else:
            self.env = ToricGameDynamicEnv(settings)

        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every 1000 steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
        
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
        if self.initialisation_settings['mask_actions']:
            self.model=MaskablePPO.load(f"trained_models/ppo_{load_model_path}")
        else:
            self.model=PPO.load(f"trained_models/ppo_{load_model_path}")
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
        #plt.show()


    def render(self, obs0_k,evaluation_settings, actions_k, initial_flips_k):
        size=evaluation_settings['board_size']
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]


        fig, (ax3,ax1,ax2) = plt.subplots(1,3, figsize=(15,5))
        a=1/(2*size)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax1.add_patch(plaq)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax2.add_patch(plaq)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax3.add_patch(plaq)

        # Draw lattice
        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax1.add_patch(lattice)

        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax2.add_patch(lattice)

        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax3.add_patch(lattice)

        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc1='darkgrey'
            if i in list(actions_k[:,0]):
                fc1 = 'darkblue'


            circle1 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc1)
            ax1.add_patch(circle1)
            ax1.annotate(str(i), pos, fontsize=8, ha="center")
        
        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc2='darkgrey'
            if i in list(actions_k[:,1]):
                fc2 = 'red'


            circle2 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc2)
            ax2.add_patch(circle2)
            ax2.annotate(str(i), pos, fontsize=8, ha="center")

        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc3='darkgrey'
            if p in list(initial_flips_k)[0]:
                fc3 = 'magenta'


            circle3 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc3)
            ax3.add_patch(circle3)
            ax3.annotate(str(i), pos, fontsize=8, ha="center")

        ax1.set_xlim([-.1,1.1])
        ax1.set_ylim([-.1,1.1])
        ax1.set_aspect(1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("actions agent")
        ax2.set_xlim([-.1,1.1])
        ax2.set_ylim([-.1,1.1])
        ax2.set_aspect(1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("actions MWPM")
        ax1.axis('off')
        ax2.axis('off')
        ax3.set_xlim([-.1,1.1])
        ax3.set_ylim([-.1,1.1])
        ax3.set_aspect(1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("initial qubit flips")
        ax3.axis('off')
        plt.show()

    def evaluate_model(self, evaluation_settings, render, number_evaluations, max_moves, check_fails):
        print("evaluating the model...")
        moves=0
        #success_MWPM=0
        #logical_errors_MWPM=0
        #observations=np.zeros((number_evaluations, evaluation_settings['board_size']*evaluation_settings['board_size']))
        #results=np.zeros((number_evaluations,2)) #1st column for agent, 2nd column for MWPM decoder
        actions=np.zeros((number_evaluations,max_moves,1)) #1st column for agent (3rd dimension)
        actions[:,:,:]=np.nan
        reward_agent=[]
        #data_evaluations=np.zeros((number_evaluations))

        for k in range(number_evaluations):
            rewards=0
            obs, info = self.env.reset()
            #initial_flips = AgentPPO.env.initial_qubits_flips

            #obs0=obs
            #observations[k,:]=obs
            #obs0_k=obs0.reshape((evaluation_settings['board_size'],evaluation_settings['board_size']))

            #MWPM_check, MWPM_actions = self.decode_MWPM_method(obs0_k, initial_flips, evaluation_settings)

            #actions[k,:MWPM_actions.shape[0],1] = MWPM_actions[:,0]

            #if MWPM_check==True:
                #success_MWPM+=1
                #results[k,1]=1 #1 for success
            #if MWPM_check==False:
                #logical_errors_MWPM+=1
                #results[k,1]=0 #0 for fail
            #self.env.render()
            if render:
                self.env.render()
            for i in range(max_moves):
                if i == (max_moves-1):
                    print("max moves/max reward reached")
                if evaluation_settings['mask_actions']:
                    action_masks=get_action_masks(self.env)
                    #print(f"{action_masks=}")
                    action, _state = self.model.predict(obs, action_masks=action_masks)
                    #print(f"{action=}")
                else:
                    action, _state = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)#, without_illegal_actions=True)
                #print(f"{done=}")
                #print(f"{reward=}")
                actions[k,i,0]=action
                moves+=1
                rewards+=reward
                if render:
                    print(info['message'])
                    self.env.render()
                if done:
                    reward_agent.append(rewards)
                    
                    #if reward == evaluation_settings['l_reward']:
                        #if check_fails:
                            #if results[k,0]==0 and results[k,1]==1:

                                #print(info['message'])

                                #self.render(obs0_k,evaluation_settings, actions[k,:,:], initial_flips)
                        #logical_errors+=1
                        #results[k,0]=0 #0 for fail
                    #if reward == evaluation_settings['s_reward']:
                        #success+=1
                        #results[k,0]=1 #1 for success
                        #self.env.render()
                        #self.env.reset()
                    break

        mean_reward=np.mean(reward_agent)
        print(f"mean reward per evaluation is {mean_reward}")
                    
            
        print(f"mean number of moves per evaluation is {moves/number_evaluations}")
        

        print("evaluation done")

        return mean_reward, actions



    def evaluate_fixed_errors(self, evaluation_settings, N_evaluates, render, number_evaluations, max_moves,check_fails, save_files):
        
        rewards_agent=[]


        for N_evaluate in N_evaluates:
            print(f"{N_evaluate=}")
            evaluation_settings['fixed'] = evaluate_fixed
            evaluation_settings['N']=N_evaluate
            self.change_environment_settings(evaluation_settings)
            reward_agent, actions = self.evaluate_model(evaluation_settings, render, number_evaluations, max_moves,check_fails)
            rewards_agent.append(reward_agent)

        rewards_agent=np.array(rewards_agent)



        evaluation_path =''
        for key, value in evaluation_settings.items():
            evaluation_path+=f"{key}={value}"

        if save_files:
            if fixed:
                np.savetxt(f"Files_results/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", rewards_agent)
                #np.savetxt(f"Files_results/rewards_dynamic_agent/actions_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions)

            else:
                np.savetxt(f"Files_results/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", rewards_agent)
                #np.savetxt(f"Files_results/rewards_dynamic_agent/actions_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions)


        return rewards_agent, actions
    

    def evaluate_error_rates(self,evaluation_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files, fixed):
        success_rates=[]
        success_rates_MWPM=[]
        observations_all=[]

        for error_rate in error_rates:
            #SET SETTINGS TO EVALUATE LOADED AGENT ON
            print(f"{error_rate=}")
            evaluation_settings['error_rate'] = error_rate
            evaluation_settings['fixed'] = evaluate_fixed

            self.change_environment_settings(evaluation_settings)
            success_rate, success_rate_MWPM, observations, results, actions = self.evaluate_model(evaluation_settings, render, number_evaluations, max_moves, check_fails)
            success_rates.append(success_rate)
            success_rates_MWPM.append(success_rate_MWPM)
            observations_all.append(observations)
            print(f"{success_rate=}")
            print(f"{success_rate_MWPM=}")



        success_rates=np.array(success_rates)
        success_rates_MWPM=np.array(success_rates_MWPM)
        observations_all=np.array(observations_all)



        evaluation_path =''
        for key, value in evaluation_settings.items():
            evaluation_path+=f"{key}={value}"

        if save_files:
            if fixed:
                np.savetxt(f"Files_results/success_rates_agent/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates)
                np.savetxt(f"Files_results/success_rates_MWPM/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates_MWPM)
                np.savetxt(f"Files_results/observations/observations_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", observations)
                np.savetxt(f"Files_results/results_agent_MWPM/results_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", results)
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_agent_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,0])
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_MWPM_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,1])
            else:
                np.savetxt(f"Files_results/success_rates_agent/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates)
                np.savetxt(f"Files_results/success_rates_MWPM/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates_MWPM)
                np.savetxt(f"Files_results/observations/observations_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", observations)
                np.savetxt(f"Files_results/results_agent_MWPM/results_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", results)
                np.savetxt(f"Files_results/actions_agent_MPWM/actions_agent_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,0])
                np.savetxt(f"Files_results/actions_agent_MPWM/actions_MWPM_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,1])

        return success_rates, success_rates_MWPM,observations, results, actions



#SETTINGS FOR RUNNING THIS SCRIPT
train=False
curriculum=False #if set to True the agent will train on N_curriculum or error_rate_curriculum examples, using the training experience from 
benchmark_MWPM=False
save_files=True
render=True
number_evaluations=1000
max_moves=200
evaluate=True
check_fails=False

board_size=5
error_rate=0.01
ent_coef=0.0
total_timesteps=50000
random_error_distribution=True
mask_actions=True #if set to True action masking is enabled, the illegal actions are masked out by the model. If set to False the agent gets a reward 'illegal_action_reward' when choosing an illegal action.
log = True #if set to True the learning curve during training is registered and saved.
lambda_value=1
fixed=True #if set to True the agent is trained on training examples with a fixed amount of N initial errors. If set to False the agent is trained on training examples given an error rate error_rate for each qubit to have a chance to be flipped.
evaluate_fixed=True #if set to True the trained model is evaluated on examples with a fixed amount of N initial errors. If set to False the trained model is evaluated on examples in which each qubit is flipped with a chance of error_rate.
#N_evaluates = [1, 2, 3,4, 5] #the number of fixed initial flips N the agent is evaluated on if evaluate_fixed is set to True
N_evaluates=[1] 
N=1 #the number of fixed initinal flips N the agent model is trained on or loaded when fixed is set to True
error_rates_eval=list(np.linspace(0.01,0.20,6))
#N_curriculums=[2,3,4]
#N_curriculums=[4]
N_curriculums=[1]

#SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'lr':0.001,
            'total_timesteps': total_timesteps,
            'random_error_distr': random_error_distribution,
            'mask_actions': mask_actions,
            'lambda': lambda_value,
            'fixed':fixed,
            'N':N#,
            #'ent':ent_coef
            }

#SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'lr':0.001,
            'total_timesteps': total_timesteps,
            'random_error_distr': random_error_distribution,
            'mask_actions': mask_actions,
            'lambda': lambda_value,
            'fixed':fixed,
            'N':N#,
            #'ent':ent_coef
            }

evaluation_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'lr':0.001,
            'total_timesteps': total_timesteps,
            'random_error_distr': random_error_distribution,
            'mask_actions': mask_actions,
            'lambda': lambda_value,
            'fixed':fixed,
            'N':N#,
            #'ent':ent_coef
            }



rewards_agent_all=[]


error_rates_curriculum=list(np.linspace(0.01,0.20,6))[1:]

#N_curriculums=[2,3,4]
#N_curriculums=[4]


#for error_rate_curriculum in error_rates_curriculum:
if fixed:
    curriculums=N_curriculums
else:
    curriculums=error_rates_curriculum


for curriculum_val in curriculums:
    
    if (train==True) and (curriculum == False) and(curriculums.index(curriculum_val)>0):
        train=False
        curriculum=True
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
        print(f"{loaded_model_settings['N']=}")
        AgentPPO.load_model(load_model_path=load_model_path)
        

    if curriculum:
        if fixed:
        
            print(f"N_curriculum = {curriculum_val}")
            initialisation_settings['N']=curriculum_val
        else:
            print(f"error_rate_curriculum={curriculum_val}")
            initialisation_settings['error_rate']=curriculum_val


        save_model_path =''
        for key, value in initialisation_settings.items():
            save_model_path+=f"{key}={value}"

        AgentPPO.change_environment_settings(initialisation_settings)

        AgentPPO.train_model(save_model_path=save_model_path)
        
        if fixed:
            loaded_model_settings['N']=curriculum_val
        else:
            loaded_model_settings['error_rate']=curriculum_val


            

    p_start = 0.01 
    p_end = 0.20
    error_rates = np.linspace(p_start,p_end,6)



    if evaluate:

        if evaluate_fixed:
            rewards_agent, actions = AgentPPO.evaluate_fixed_errors(evaluation_settings, N_evaluates, render, number_evaluations, max_moves, check_fails, save_files)
        else:
            success_rates, success_rates_MWPM,observations, results, actions = AgentPPO.evaluate_error_rates(evaluation_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files, fixed)


    rewards_agent_all.append(rewards_agent)

rewards_agent_all=np.array(rewards_agent_all)


evaluation_path =''
for key, value in evaluation_settings.items():
    evaluation_path+=f"{key}={value}"


if fixed:
    path = f"Figure_results/Results_benchmarks/results_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.pdf"
else:
    path = f"Figure_results/Results_benchmarks/results_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"

print(f"{rewards_agent_all}")

plt.figure()
#for i in len(N_curriculums):
plt.plot(N_evaluates,rewards_agent_all, label=f"N={N_curriculums}")
plt.title("PPO agent on dynamic environment")
plt.legend()
plt.xlabel("N")
plt.ylabel("Reward")
plt.savefig(path)





#error_rates=np.array(N_evaluates)/50
#error_rates=list(error_rates)



#if benchmark_MWPM:
    #sim_data, sim_all_data = simulate(simulation_settings)
    #plot(plot_settings, sim_data, sim_all_data, success_rates_all, error_rates, N_curriculums)

#if plot_illegal_actions_rate:


#if fixed:
    #path_plot = f"Figure_results/Results_benchmarks/PPO_vs_MWPM_{evaluation_path}_{loaded_model_settings['N']}.pdf"
#else:
    #path_plot = f"Figure_results/Results_benchmarks/PPO_vs_MWPM_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"


#plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM, error_rates_eval, board_size,path_plot,loaded_model_settings['N'], loaded_model_settings['error_rate'],evaluate_fixed)