import numpy as np
import os


import neat
import config
from toric_game_env import ToricGameEnv
from game import ToricCodeGame
from config import GameMode, RewardMode, ErrorModel
from perspectives import Perspectives


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
#qubit_number=[8,13,10,9,2, 17]
#qubit_number = np.random.randint(0,18,5)
#print(f"qubit_numbers are: {qubit_number}")

game=ToricCodeGame(config.get_default_config())
env = ToricGameEnv(board_size, error_model, channels, config.get_default_config()['Training']['memory'])
q = env.state.qubit_pos
done=False
#env.generate_errors(error_rate)
#env.render()
#qubit_number = np.random.randint(0,18,3)
current_state = env.generate_errors(error_rate)
#while len(env.state.syndrome_pos)!=2:
#    qubit_number = np.random.randint(0,18,5)
#    current_state = env.generate_errors_test(error_rate)
#    print(len(env.state.syndrome_pos))

print(f"current reward: {env.reward}")

env.render()

while not done:
    redo_qubit = input("Give qubit number to redo the errors: ").split(',')
    redo_qubit = [eval(i) for i in redo_qubit]
    #current_state = env.redo_errors_test(redo_qubits)
    img, reward, done, current_state = env.step(q[redo_qubit[0]], pauli_opt, without_illegal_actions=False)
    #current_state =env.step(q[redo_qubit[0]], pauli_opt, without_illegal_actions=False)
    print(f"current reward: {reward}")
    print(current_state)
    env.render()

