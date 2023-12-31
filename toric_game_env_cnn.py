from __future__ import division
from typing import Any
import numpy as np

import gymnasium as gym
#import gym
from gymnasium.utils import seeding
from gymnasium import spaces
import gymnasium
import sys, os
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from config import ErrorModel

### Environment
class ToricGameEnv(gym.Env):
    '''
    ToricGameEnv environment. Effective single player game.
    '''

    def __init__(self, settings):
        """
        Args:
            opponent: Fixed
            board_size: board_size of the board to use
        """

        self.settings=settings

        self.board_size = settings['board_size']
        self.error_model = settings['error_model']
        self.channels = [0]
        self.memory = False
        self.error_rate = settings['error_rate']
        self.logical_error_reward=settings['l_reward']
        self.continue_reward=settings['c_reward']
        self.success_reward=settings['s_reward']
        self.mask_actions=settings['mask_actions']
        self.illegal_action_reward = settings['i_reward']
        self.lambda_value = settings['lambda']
        self.N = settings['N']

        # Keep track of the moves
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]


        # Empty State
        self.state = Board(self.board_size)
        self.done = None

        #self.observation_space=spaces.MultiBinary(25)
        self.observation_space = spaces.Box(low=0, high=1,shape=(1,self.board_size,self.board_size), dtype=np.uint8) #3x3 plaquettes on which we can view syndromes
        print(self.observation_space.sample())
        self.action_space = spaces.discrete.Discrete(len(self.state.qubit_pos)) #0-17 qubits on which a bit-flip error can be introduced


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

    def action_masks(self):
        #self.action_masks_list = []
        #action_mask=[]
        self.mask_qubits=[]
        self.action_masks_list=np.zeros((len(self.state.qubit_pos)))
        self.action_masks_list[:]=False
        for i in self.state.syndrome_pos:
            a,b = i[0],i[1]
            mask_coords = [[(a-1)%(2*self.state.size),b%(2*self.state.size)],[a%(2*self.state.size),(b-1)%(2*self.state.size)],[a%(2*self.state.size),(b+1)%(2*self.state.size)],[(a+1)%(2*self.state.size),b%(2*self.state.size)]]
            for j in mask_coords:
                qubit_number = self.state.qubit_pos.index(j)
                self.mask_qubits.append(qubit_number)
                self.action_masks_list[qubit_number]=True
                #action_mask.append(qubit_number)
        '''
        action_mask = list(set(action_mask))
        for i in range(len(self.state.qubit_pos)):
            if i in action_mask:
                self.action_masks_list.append(True)
            else:
                self.action_masks_list.append(False)
        '''


        self.action_masks_list=list(self.action_masks_list)


        return self.action_masks_list
        #self.mask_qubits.append(qubit_number)
        #self.action_mask = list(set(self.action_mask))
        #print(f" mask action space on qubits = {self.mask_qubits}")





    def generate_errors(self):
        # Reset the board state
        self.state.reset()

        # Let the opponent do it's initial evil
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]

        self._set_initial_errors()
        #self.render()
        #print(f"syndrome pos = {self.state.syndrome_pos[0]}")
        self.action_masks_list=self.action_masks()

        self.done = self.state.is_terminal()
        self.reward = 0
        if self.done:
            self.reward = self.success_reward
            if self.state.has_logical_error(self.initial_qubits_flips):
                self.reward = self.logical_error_reward

        return self.state.encode(self.channels, self.memory)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
         super().reset(seed=seed, options=options)
         initial_observation = self.generate_errors()
         return initial_observation, {'state': self.state, 'message':"reset"}


    def close(self):
        self.state = None

    def render(self, mode="human", close=False):
        fig, ax = plt.subplots()
        a=1/(2*self.board_size)

        for i, p in enumerate(self.state.plaquet_pos):
            if self.state.op_values[0][i]==1:
                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        for i, p in enumerate(self.state.star_pos):
            if self.state.op_values[1][i]==1:
                fc = 'green'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        # Draw lattice
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax.add_patch(lattice)

        for i, p in enumerate(self.state.qubit_pos):
            pos=(a*p[0], a*p[1])
            fc='darkgrey'
            if self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 0:
                fc='darkblue'
            elif self.state.qubit_values[0][i] == 0 and self.state.qubit_values[1][i] == 1:
                fc='darkred'
            elif self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 1:
                fc='darkmagenta'
            circle = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc)
            ax.add_patch(circle)
            plt.annotate(str(i), pos, fontsize=8, ha="center")

        ax.set_xlim([-.1,1.1])
        ax.set_ylim([-.1,1.1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

    def step(self, location,  without_illegal_actions=True):
        
        '''
        Args:
            location: coord of the qubit to flip
            operator: Pauli matrix to apply
        Return:
            observation: board encoding,
            reward: reward of the game,
            done: boolean,
            info: state dict
        '''
        # If already terminal, then don't do anything, count as win
        if self.done:
            self.reward = self.success_reward
            return self.state.encode(self.channels, self.memory), self.success_reward, True, False,{'state': self.state, 'message':"success"}

        # Check if we flipped twice the same qubit 
        #pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
        #pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)
        pauli_opt=0
        pauli_X_flip=True
        pauli_Z_flip=False
        #self.render()


        if self.mask_actions==False:

            self.action_masks_list = self.action_masks()

            if location not in self.mask_qubits:
                return self.state.encode(self.channels, self.memory), self.illegal_action_reward, True, False,{'state': self.state, 'message': "illegal_action"}


        if not without_illegal_actions:
            if pauli_X_flip and location in self.qubits_flips[0]:
                return self.state.encode(self.channels, self.memory), self.illegal_action_reward, True, False,{'state': self.state, 'message': "illegal_action"}
            if pauli_Z_flip and location in self.qubits_flips[1]:
                return self.state.encode(self.channels, self.memory), self.illegal_action_reward, True, False,{'state': self.state, 'message': "illegal_action"}
        


        if pauli_X_flip:
            self.qubits_flips[0].append(location)

        if pauli_Z_flip:
            self.qubits_flips[1].append(location)


        self.state.act(self.state.qubit_pos[location], pauli_opt)

        #print(self.initial_qubits_flips[0][0])
        

        # Reward: if nonterminal, then the reward is 0
        if not self.state.is_terminal():
            self.done = False
            #if len(self.qubits_flips[0])<=len(self.initial_qubits_flips[0]):

            return self.state.encode(self.channels, self.memory), self.continue_reward, False, False,{'state': self.state, 'message':"continue"}
    
            #else:
                #return self.state.encode(self.channels, self.memory), self.continue_reward*10, False, False,{'state': self.state, 'message':"continue"}
        
        # We're in a terminal state. Reward is 1 if won, -1 if lost

        self.done = True
        if self.state.has_logical_error(self.initial_qubits_flips):
            return self.state.encode(self.channels, self.memory), self.logical_error_reward, True, False,{'state': self.state, 'message':"logical_error"}
        else:
            return self.state.encode(self.channels, self.memory), self.success_reward, True, False,{'state': self.state, 'message':"success"}


    def generate_new_error(self):

        q = np.random.randint(0,len(self.state.qubit_pos))
        q = self.state.qubit_pos[q]

        pauli_opt=0
        pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
        pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)

        if pauli_X_flip:
            self.initial_qubits_flips[0].append( q )
        if pauli_Z_flip:
            self.initial_qubits_flips[1].append( q )

        self.state.act(q, pauli_opt)

        #self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))


    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilitic mode
        # Pick random sites according to error rate
        for q in self.state.qubit_pos:    
            if np.random.rand() < self.error_rate:
            #print(f" qubit has bit flip error on {self.state.qubit_pos.index(q)}")
                if self.error_model == ErrorModel["UNCORRELATED"]:
                    pauli_opt = 0
                elif self.error_model == ErrorModel["DEPOLARIZING"]:
                    pauli_opt = np.random.randint(0,3)

                pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
                pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


                if pauli_X_flip:
                    self.initial_qubits_flips[0].append( q )
                if pauli_Z_flip:
                    self.initial_qubits_flips[1].append( q )

                self.state.act(q, pauli_opt)


        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))

class ToricGameEnvFixedErrs(ToricGameEnv):
    def __init__(self, settings):
        super().__init__(settings)
        

    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilistic mode
        #self.N=np.random.randint(1,4)
        for q in np.random.choice(len(self.state.qubit_pos), self.N, replace=False): 
        #for q in [38, 39]:
            #q = 23
            #q = np.random.choice([21, 15, 46, 19, 25, 26, 14],1)[0] 
            q = self.state.qubit_pos[q]
            #print(f" qubit has bit flip error on {self.state.qubit_pos.index(q)}")
            if self.error_model == ErrorModel["UNCORRELATED"]:
                pauli_opt = 0
            elif self.error_model == ErrorModel["DEPOLARIZING"]:
                pauli_opt = np.random.randint(0,3)

            pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
            pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


            if pauli_X_flip:
                self.initial_qubits_flips[0].append( q )
            if pauli_Z_flip:
                self.initial_qubits_flips[1].append( q )

            self.state.act(q, pauli_opt)
            #break
        # Now unflip the qubits, they're a secret

        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))

class ToricGameEnvLocalErrs(ToricGameEnv):
    def __init__(self, settings):
        super().__init__(settings)
        

    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilitic mode

        #probability_matrix = np.random.rand(len(self.state.qubit_pos))

        #first choose the first qubit to flip
        for index_qubit in range(len(self.state.qubit_pos)):
            if np.random.rand() < self.error_rate:

                q = self.state.qubit_pos[index_qubit]
                #print(f" qubit has bit flip error on {self.state.qubit_pos.index(q)}")
                if self.error_model == ErrorModel["UNCORRELATED"]:
                    pauli_opt = 0
                elif self.error_model == ErrorModel["DEPOLARIZING"]:
                    pauli_opt = np.random.randint(0,3)

                pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
                pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


                if pauli_X_flip:
                    self.initial_qubits_flips[0].append( q )
                if pauli_Z_flip:
                    self.initial_qubits_flips[1].append( q )

                self.state.act(q, pauli_opt)

                dist_matrix = distance_matrix(self.state.qubit_pos, self.state.qubit_pos) #1x aanmaken
                selected_dist_matrix = dist_matrix[index_qubit]/np.sum(dist_matrix[index_qubit])
                weighted_matrix = (1-selected_dist_matrix)
                where_zero = np.argwhere(weighted_matrix == np.max(weighted_matrix))[0]
                weighted_matrix[where_zero] = 0

                if np.random.rand() < self.error_rate:
                    #for index_second_qubit in range(len(self.state.qubit_pos)): #permute
                    probabilities = np.exp(-weighted_matrix/self.lambda_value)
                    probabilities = probabilities/np.sum(probabilities)
                    index_second_qubit = np.random.choice(range(len(self.state.qubit_pos)), p=probabilities)
                    # for ind, secondbit in enumerate(self.state.qubit_pos):
                    # if np.random.rand()*np.exp(-weighted_matrix[index_second_qubit]/self.lambda_value) < self.error_rate:
                    s = self.state.qubit_pos[index_second_qubit]
                    #print(f" qubit has bit flip error on {self.state.qubit_pos.index(q)}")
                    if self.error_model == ErrorModel["UNCORRELATED"]:
                        pauli_opt = 0
                    elif self.error_model == ErrorModel["DEPOLARIZING"]:
                        pauli_opt = np.random.randint(0,3)

                    pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
                    pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


                    if pauli_X_flip:
                        self.initial_qubits_flips[0].append( s )
                    if pauli_Z_flip:
                        self.initial_qubits_flips[1].append( s )

                    self.state.act(s, pauli_opt)
                    break


        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))

class ToricGameEnvLocalFixedErrs(ToricGameEnv):
    def __init__(self, settings):
        super().__init__(settings)
        


    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''

        #first choose the first qubit to flip
        for index_qubit in np.random.choice(len(self.state.qubit_pos), self.N, replace=False):


            q = self.state.qubit_pos[index_qubit]
            #print(f" qubit has bit flip error on {self.state.qubit_pos.index(q)}")
            if self.error_model == ErrorModel["UNCORRELATED"]:
                pauli_opt = 0
            elif self.error_model == ErrorModel["DEPOLARIZING"]:
                pauli_opt = np.random.randint(0,3)

            pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
            pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


            if pauli_X_flip:
                self.initial_qubits_flips[0].append( q )
            if pauli_Z_flip:
                self.initial_qubits_flips[1].append( q )

            self.state.act(q, pauli_opt)

            dist_matrix = distance_matrix(self.state.qubit_pos, self.state.qubit_pos)
            selected_dist_matrix = dist_matrix[index_qubit]/np.sum(dist_matrix[index_qubit])
            weighted_matrix = (1-selected_dist_matrix)
            where_zero = np.argwhere(weighted_matrix == np.max(weighted_matrix))[0]
            weighted_matrix[where_zero] = 0
            
            if (self.N%2==0):
                
                probabilities = np.exp(-weighted_matrix/self.lambda_value)
                probabilities = probabilities/np.sum(probabilities)
                index_second_qubit = np.random.choice(range(len(self.state.qubit_pos)), p=probabilities)
                
                # if np.random.rand()*np.exp(-weighted_matrix[index_second_qubit]/self.lambda_value) < self.error_rate:
                s = self.state.qubit_pos[index_second_qubit]
                #print(f" qubit has bit flip error on {self.state.qubit_pos.index(q)}")
                if self.error_model == ErrorModel["UNCORRELATED"]:
                    pauli_opt = 0
                elif self.error_model == ErrorModel["DEPOLARIZING"]:
                    pauli_opt = np.random.randint(0,3)

                pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
                pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


                if pauli_X_flip:
                    self.initial_qubits_flips[0].append( s )
                if pauli_Z_flip:
                    self.initial_qubits_flips[1].append( s )
                

                self.state.act(s, pauli_opt)



class Board(object):
    '''
    Basic Implementation of a ToricGame Board, actions are int [0,2*board_size**2)
    o : qubit
    P : plaquette operator
    x : star operator

    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |

    '''
    @staticmethod
    def component_positions(size):
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]
        star_pos    = [[x,y] for x in range(0,2*size,2) for y in range(0,2*size,2)]

        return qubit_pos, plaquet_pos, star_pos

    def __init__(self, board_size):
        self.size = board_size

        # Real-space locations
        self.qubit_pos, self.plaquet_pos, self.star_pos  = self.component_positions(self.size)

        # Mapping between 1-index and 2D position
        '''
        self.qubit_dict, self.star_dict, self.plaquet_dict = {},{},{}
        for i in range(2*self.size*self.size):
            self.qubit_dict[self.qubit_pos[i]] = i
        for i in range(self.size*self.size):
            self.star_dict[i] = self.star_pos[i]
            self.plaquet_dict[i] = self.plaquet_pos[i]
        '''

        # Define here the logical error for efficiency
        self.z1pos = [[0,x] for x in range(1, 2*self.size, 2)]
        self.z2pos = [[y,0] for y in range(1, 2*self.size, 2)]
        self.x1pos = [[1,x] for x in range(0, 2*self.size, 2)]
        self.x2pos = [[y,1] for y in range(0, 2*self.size, 2)]

        self.reset()

    def reset(self):
        #self.board_state = np.zeros( (2, 2*self.size, 2*self.size) )
        
        self.qubit_values = np.zeros((2, 2*self.size*self.size))
        self.op_values = np.zeros((2, self.size*self.size))

        self.syndrome_pos = [] # Location of syndromes

    def act(self, coord, operator):
        '''
            Args: input action in the form of position [x,y]
            coord: real-space location of the qubit to flip
        '''

        pauli_X_flip = (operator==0 or operator==2)
        pauli_Z_flip = (operator==1 or operator==2)



        #qubit_index = self.qubit_pos.index(coord)
        qubit_index=self.qubit_pos.index(coord)
        #coord = self.qubit_pos[coord]

        # Flip it!
        if pauli_X_flip:
            self.qubit_values[0][qubit_index] = (self.qubit_values[0][qubit_index] + 1) % 2
        if pauli_Z_flip:
            self.qubit_values[1][qubit_index] = (self.qubit_values[1][qubit_index] + 1) % 2

        # Update the syndrome measurements
        # Only need to incrementally change
        # Find plaquettes that the flipped qubit is a part of
        plaqs=[]
        if pauli_X_flip:
            if coord[0] % 2 == 0:
                plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]
            else:
                plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]

        if pauli_Z_flip:
            if coord[0] % 2 == 0:
                plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]
            else:
                plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]


        # Update syndrome positions
        for plaq in plaqs:
            if plaq in self.syndrome_pos:
                self.syndrome_pos.remove(plaq)
            else:
                self.syndrome_pos.append(plaq)

            # The plaquette or vertex operators are only 0 or 1
            if plaq in self.star_pos:
                op_index = self.star_pos.index(plaq)
                channel = 1
            elif plaq in self.plaquet_pos:
                op_index = self.plaquet_pos.index(plaq)
                channel = 0

        
            self.op_values[channel][op_index] = (self.op_values[channel][op_index] + 1) % 2





    def is_terminal(self):
        # Not needed I think
        #if len(self.get_legal_action()) == 0:
        #    return True

        # Are all syndromes removed?
        return len(self.syndrome_pos) == 0

    def has_logical_error(self, initialmoves, debug=False):
        if debug:
            print("Initial errors:", [self.qubit_pos.index(q) for q in initialmoves])



        '''
        #check whether initial moves occur an even amount of times -> these should be removed as these result in an un-flipped qubit!
        for i in initialmoves[0]:
            counter = initialmoves[0].count(i)
            if (counter%2==0):
                initialmoves[0]=list(filter(lambda a: a!=i, initialmoves[0])) #KLOPT NOG NIET!
        '''

        # Check for Z logical error
        zerrors = [0,0]
        for pos in self.z1pos:
            if pos in initialmoves[0]:
                zerrors[0] += 1
            qubit_index = self.qubit_pos.index(pos)
            zerrors[0] += self.qubit_values[0][ qubit_index ]

        for pos in self.z2pos:
            if pos in initialmoves[0]:
                zerrors[1] += 1
            qubit_index = self.qubit_pos.index(pos)
            zerrors[1] += self.qubit_values[0][ qubit_index ]

        # Check for X logical error
        xerrors = [0,0]
        for pos in self.x1pos:
            if pos in initialmoves[1]:
                xerrors[0] += 1
            qubit_index = self.qubit_pos.index(pos)
            xerrors[0] += self.qubit_values[1][ qubit_index ]

        for pos in self.x2pos:
            if pos in initialmoves[1]:
                xerrors[1] += 1
            qubit_index = self.qubit_pos.index(pos)
            xerrors[1] += self.qubit_values[1][ qubit_index ]

        #print("Zerrors", zerrors)


        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1) or \
            (xerrors[0]%2 == 1) or (xerrors[1]%2 == 1):
            return True

        return False


    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        return f"Qubit Values: {self.qubit_values}, Operator values: {self.op_values}"

    def encode(self, channels, memory):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        # In case of uncorrelated noise for instance, we don't need information
        # about the star operators

        img=np.array([],dtype=np.uint8)
        for channel in channels:
            img = np.concatenate((img, self.op_values[channel]))
            if memory:
                img = np.concatenate((img, self.qubit_values[channel]))

        return np.reshape(img, newshape=(1,self.size,self.size))

    def image_view(self, number=False, channel=0):
        image = np.empty((2*self.size, 2*self.size), dtype=object)
        #print(image)
        for i, plaq in enumerate(self.plaquet_pos):
            if self.op_values[0][i] == 1:
                image[plaq[0], plaq[1]] = "P"+str(i) if number else "P"
            elif self.op_values[0][i] == 0:
                image[plaq[0], plaq[1]] = "x"+str(i) if number else "x"
        for i,plaq in enumerate(self.star_pos):
            if self.op_values[1][i] == 1:
                image[plaq[0], plaq[1]] = "S"+str(i) if number else "S"
            elif self.op_values[1][i] == 0:
                image[plaq[0], plaq[1]] = "+"+str(i) if number else "+"

        for i,pos in enumerate(self.qubit_pos):
            image[pos[0], pos[1]] = str(int(self.qubit_values[channel,i]))+str(i) if number else str(int(self.qubit_values[channel, i]))

        return np.array(image)

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    initialisation_settings = {'board_size': 5,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': 1,
            'l_reward': 1,
            's_reward': 1,
            'c_reward':1,
            'i_reward':1,
            'lr':1,
            'total_timesteps': 1,
            'random_error_distr': True,
            'mask_actions': 1,
            'lambda': 1,
            'fixed':1,
            'N':1#,
            #'ent':ent_coef
            }
    check_env(ToricGameEnv(settings=initialisation_settings))