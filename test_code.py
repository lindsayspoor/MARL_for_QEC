import numpy as np
from scipy.spatial import distance_matrix

def make_grids(L):
    """TESTED: generate a grid of size LxL
    input:
            L = size of grid
    output:
            stab = LxL grid of stabilizers with 0(no error)
                qubits are between stabilizers
            qubits = 2LxL grid of qubits
            """
    stab = [[0 for col in range(L)] for row in range(L)]
    qubits = [[0 for col in range(L)] for row in range(2 * L)]
    return stab, np.array(qubits)

def qubit_positions(L):
    qubit_pos=[[x,y] for x in range(2*L) for y in range((x+1)%2, 2*L, 2)]
    qubit_pos=np.array(qubit_pos).reshape(2*L,L,2)
    return qubit_pos

grid_s, grid_q = make_grids(3)
qubit_pos=qubit_positions(3)
qubit=qubit_pos[3,2]
qubit_pos_new=qubit_pos.reshape(2*3*3,2)
idx_x = np.argwhere(qubit_pos_new[:,0]==qubit[0])
idx_y=np.argwhere(qubit_pos_new[idx_x,1]==qubit[1])
index_qubit=idx_x[idx_y[0][0]][0]
print(index_qubit)


dist_matrix=distance_matrix(qubit_pos_new, qubit_pos_new)
selected_dist_matrix=dist_matrix[index_qubit]/np.sum(dist_matrix[index_qubit])
print(selected_dist_matrix)
weighted_matrix = (1-selected_dist_matrix)
print(weighted_matrix)
where_zero = np.argwhere(weighted_matrix == np.max(weighted_matrix))[0]
weighted_matrix[where_zero] = 0

print(weighted_matrix)
                
