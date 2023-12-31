# The basis to simulate the toric code: generate grids, simulate errors, check if correction is complete etc

from random import random
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
    return stab, qubits


def print_grid_stab(grid):
    """TESTED: print the input grid per row, stabilizers """
    print('-' * len(grid[0]))
    for row in grid:
        st = ' '.join([str(n) for n in row])
        print(st)
    print('-' * len(grid[0]))


def print_grid_qubits(grid):
    """TESTED: print the input grid per row, qubits"""
    print('-' * len(grid[0]) * 3)

    for i in range(len(grid)):
        if i % 2 == 0:
            print('+ ' + ' + '.join([str(x) for x in grid[i]]))
        else:
            print('   '.join([str(x) for x in grid[i]]) + '  ')
    print('-' * len(grid[0]) * 3)


def generate_error(grid_s, grid_q, px):
    """TESTED: Generate random errors on qubits, and adds 1 to the stabilizers
    if an error on neighbouring qubit occured
    Input:
            grid: LxL grid of stabilizers
            px: probability to have an error on a qubits
    Output:
            grid_s: LxL grid of stabilizers, with values 0-4 for 0-4 errors
                on neighbouring qubits
            grid_q: 2LxL grid of qubits, with values 0-1 for 0 or 1 error on the qubits
            """
    # loop through all qubits:
    for row_idx in range(len(grid_q)):
        for col_idx in range(len(grid_q[0])):
            error = random() <= px
            if not error:
                # nothing has to be changed
                continue
            if row_idx % 2 == 0:
                # above/under stabilizers -> same column
                stab_row = int(row_idx / 2)
                grid_s[stab_row][col_idx] += 1  # stabilizer under qubit
                grid_s[stab_row - 1][col_idx] += 1  # stabilizer above qubit
            else:
                # left/right of stabilizers -> same row
                stab_row = int((row_idx - 1) / 2)
                grid_s[stab_row][col_idx] += 1  # stabilizer right of qubit
                grid_s[stab_row][col_idx - 1] += 1  # stabilizer right of qubit
            grid_q[row_idx][col_idx] += 1
    
    return grid_s, grid_q

def qubit_positions(L):
    qubit_pos=[[x,y] for x in range(2*L) for y in range((x+1)%2, 2*L, 2)]
    qubit_pos=np.array(qubit_pos).reshape(2*L,L,2)
    return qubit_pos

def generate_local_error(grid_s, grid_q, px, L, lambda_value):
    # loop through all qubits:
    qubit_pos = qubit_positions(L)

    for row_idx in range(len(grid_q)):
        for col_idx in range(len(grid_q[0])):
            if np.random.rand() < px:
                if row_idx % 2 == 0:
                    # above/under stabilizers -> same column
                    stab_row = int(row_idx / 2)
                    grid_s[stab_row][col_idx] += 1  # stabilizer under qubit
                    grid_s[stab_row - 1][col_idx] += 1  # stabilizer above qubit
                else:
                    # left/right of stabilizers -> same row
                    stab_row = int((row_idx - 1) / 2)
                    grid_s[stab_row][col_idx] += 1  # stabilizer right of qubit
                    grid_s[stab_row][col_idx - 1] += 1  # stabilizer right of qubit
                grid_q[row_idx][col_idx] += 1

                qubit=qubit_pos[row_idx,col_idx]
                qubit_pos_new=qubit_pos.reshape(2*L*L,2)
                idx_x = np.argwhere(qubit_pos_new[:,0]==qubit[0])
                idx_y=np.argwhere(qubit_pos_new[idx_x,1]==qubit[1])
                index_qubit=idx_x[idx_y[0][0]][0]


                dist_matrix=distance_matrix(qubit_pos_new, qubit_pos_new)
                selected_dist_matrix=dist_matrix[index_qubit]/np.sum(dist_matrix[index_qubit])
                weighted_matrix = (1-selected_dist_matrix)
                where_zero = np.argwhere(weighted_matrix == np.max(weighted_matrix))[0]
                weighted_matrix[where_zero] = 0

                for row_2_idx in range(len(grid_q)):
                    for col_2_idx in range(len(grid_q[0])):
                        qubit_2=qubit_pos[row_idx,col_idx]
                        qubit_pos_new=qubit_pos.reshape(2*L*L,2)
                        idx_x_2 = np.argwhere(qubit_pos_new[:,0]==qubit_2[0])
                        idx_y_2=np.argwhere(qubit_pos_new[idx_x_2,1]==qubit_2[1])
                        index_second_qubit=idx_x[idx_y_2[0][0]][0]
                        if np.random.rand()*np.exp(-weighted_matrix[index_second_qubit]/lambda_value) < px:
                            if row_2_idx % 2 == 0:
                                # above/under stabilizers -> same column
                                stab_row = int(row_2_idx / 2)
                                grid_s[stab_row][col_2_idx] += 1  # stabilizer under qubit
                                grid_s[stab_row - 1][col_2_idx] += 1  # stabilizer above qubit
                            else:
                                # left/right of stabilizers -> same row
                                stab_row = int((row_2_idx - 1) / 2)
                                grid_s[stab_row][col_2_idx] += 1  # stabilizer right of qubit
                                grid_s[stab_row][col_2_idx - 1] += 1  # stabilizer right of qubit
                            grid_q[row_2_idx][col_2_idx] += 1
                        break


    return grid_s, grid_q


def check_correction(grid_q):
    """(tested for random ones):Check if the correction is correct(no logical X gates)
    input:
        grid_q: grid of qubit with errors and corrections
    output:
        corrected: boolean whether correction is correct.
    """
    # correct if even times logical X1,X2=> even number of times through certain edges
    # upper row = X1
    if sum(grid_q[0]) % 2 == 1:
        return (False, 'X1')
    # odd rows = X2
    if sum([grid_q[x][0] for x in range(1, len(grid_q), 2)]) == 1:
        return (False, 'X2')

    # and if all stabilizers give outcome +1 => even number of qubit flips for each stabilizer
    # is this needed? or assume given stabilizer outcome is corrected for sure?
    for row_idx in range(int(len(grid_q) / 2)):
        for col_idx in range(len(grid_q[0])):
            all_errors = 0
            all_errors += grid_q[2 * row_idx][col_idx]  # above stabilizer
            all_errors += grid_q[2 * row_idx + 1][col_idx]  # left of stabilizer
            if row_idx < int(len(grid_q) / 2) - 1:  # not the last row
                all_errors += grid_q[2 * (row_idx + 1)][col_idx]
            else:  # last row
                all_errors += grid_q[0][col_idx]
            if col_idx < len(grid_q[2 * row_idx + 1]) - 1:  # not the last column
                all_errors += grid_q[2 * row_idx + 1][col_idx + 1]
            else:  # last column
                all_errors += grid_q[2 * row_idx + 1][0]
            if all_errors % 2 == 1:
                return (False, 'stab', row_idx, col_idx)  # stabilizer gives error -1

    return (True, 'end')
    # other way of checking: for each row, look if no errors on qubits, => no loop around torus,so no gate applied.
    # and similar for columns