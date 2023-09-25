from MWPM_decoder import simulate_MWPM
#from Peeling_decoder import simulate_peeling
#from UF_decoder import simulate_UF
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def gen_px_delta(start, end, delta):  # generate px
    res = []
    current = start
    while current <= end:
        res.append(current)
        current += delta
        current = round(current, 4)  # up to .x%
    return res


def std(n_correct, n_samples):  # standard deviation for errorbars
    correct_part = n_correct * (1 - n_correct / n_samples) ** 2
    fail_part = (n_samples - n_correct) * (n_correct / n_samples) ** 2
    total = ((correct_part + fail_part) / (n_samples * (n_samples - 1))) ** 0.5
    return total


if __name__ == '__main__':
    decoder = 'MWPM'
    #decoder = 'UF'
    # decoder = 'peeling'

    #settings
    board_size = 3
    error_rates=np.linspace(0.05, 0.25, 5)
    #error_rates=[0.2]
    error_rate=0.2
    pauli_opt=0
    num_initial_errors = 3
    logical_error_reward=-1.0
    success_reward=1.0
    continue_reward=0.0
    learning_rate=0.0005
    total_timesteps=1000000
    train=True
    number_evaluations=1000
    max_moves=200
    render=False
    with_error_rates=True
    random_error_distribution = False

    # gridsizes to simulate:
    all_L = [3] #MWPM
    #all_L = [9, 17, 25, 33, 41]  # UF

    N = 1000  # number of simulations
    delta_p = 0.001  # distance between px # 0.00x = 0.x%
    #delta_p = 0.1  # distance between px # 0.00x = 0.x%
    p_start = 0.01
    p_end = 0.25
    path = 'Results_benchmarks/'
    plot_file_name = path + decoder + '_L=' + ','.join([str(x) for x in all_L]) + '_N=' + str(
        N) + 'p_start=' + str(p_start).replace('.', ',') + 'p_end=' + str(p_end).replace('.', ',') + '.pdf'
    tex_plot = False
    save_data = True
    plot_all = False  # plot all available data if True, else only data from this run
    data_filename = 'data_' + decoder + '.txt'
    # odd: threshold around 0.1
    # even: threshold around 0.12
    if tex_plot:
        file_name = 'tex' + plot_file_name

    if decoder == 'MWPM':
        sim_func = simulate_MWPM
    elif decoder == 'UF':
        sim_func = simulate_UF
    elif decoder == 'peeling':
        sim_func = simulate_peeling
    else:
        sim_func = False
    data = []
    all_px = gen_px_delta(p_start, p_end, delta_p)
    #all_px.append(0.11)
    #all_px.append(0.167)

    for L in all_L:
        print('L = ', L)
        for p_idx in range(len(all_px)):
            p = all_px[p_idx]
            if p_idx%3 == 0:
                print('progress: ', 100*round(p_idx/len(all_px),3), '%')
            k = 0
            for i in range(N):
                result = sim_func(L, p)
                if result:
                    k += 1
            if p==0.11:
                print(k)
            if p==0.167:
                print(k)
            data.append([L, p, k, N])


    if save_data:
        all_data = []
        try:
            with open(data_filename, 'r') as f:
                for line in f:
                    splitted = line.split()
                    all_data.append([int(splitted[0]), float(splitted[1]), int(splitted[2]), int(splitted[3])])
        except FileNotFoundError:
            print('File not found yet')

        # processing:
        for new_data in data:
            for old_data in all_data:
                if new_data[0] == old_data[0] and new_data[1] == old_data[1]:
                    # data point found -> add new data
                    old_data[2] += new_data[2]
                    old_data[3] += new_data[3]
                    break
            else:
                # if not found:
                all_data.append(new_data)

        with open(data_filename, 'w') as f:
            for d in all_data:
                f.write(' '.join([str(x) for x in d]))
                f.write('\n')

    # plotting:
    if plot_all:
        plot_data = all_data
    else:
        plot_data = data
    if tex_plot:
        plt.rc('text', usetex=True)
    p_x = defaultdict(list)
    p_corr = defaultdict(list)
    errorbars = defaultdict(list)
    for el in plot_data:
        L = el[0]
        p_c = el[2] / el[3]
        p_x[L].append(el[1])
        p_corr[L].append(100*p_c)
        errorbars[L].append(std(el[2], el[3]))


    plt.figure()
    for i in range(len(all_L)):
        plt.errorbar(p_x[all_L[i]], p_corr[all_L[i]], yerr=errorbars[all_L[i]],linestyle='-.', label=r'$d = ' + str(all_L[i]) + '$' + " MWPM", color='blue', linewidth = 0.5)

        success_rates_random = np.loadtxt(f"files_success_rates/success_rate_ppo_mlp_timesteps_{total_timesteps}_lr_{learning_rate}_d_{all_L[i]}_sr_{success_reward}_cr_{continue_reward}_ler_{logical_error_reward}.csv", delimiter=",")
        success_rates_random_local = np.loadtxt(f"files_success_rates/success_rates_ppo_mlp_random_local_timesteps_{total_timesteps}_lr_{learning_rate}_d_{all_L[i]}_sr_{success_reward}_cr_{continue_reward}_ler_{logical_error_reward}.csv", delimiter=",")
       
        success_rates_local = np.loadtxt(f"files_success_rates/success_rates_ppo_mlp_local_timesteps_{total_timesteps}_lr_{learning_rate}_d_{all_L[i]}_sr_{success_reward}_cr_{continue_reward}_ler_{logical_error_reward}.csv", delimiter=",")
        plt.scatter(np.linspace(0.05,0.25,5), success_rates_random*100, label=f"d={all_L[i]} PPO agent, trained & evaluated on random errors", color='darkblue', marker="^", s=30)
        plt.scatter(np.linspace(0.05,0.25,5), success_rates_local*100, label=f"d={all_L[i]} PPO agent, trained & evaluated on local errors", color='red', marker="^", s=30)
        plt.scatter(np.linspace(0.05,0.25,5), success_rates_random_local*100, label=f"d={all_L[i]} PPO agent, trained on random errors, evaluated on local errors", color='green', marker="^", s=30)
    plt.axis([p_start, p_end, 0, 100])
    plt.title(r'Toric Code - ' + decoder + f" reward scheme r_succ={success_reward}, r_fail={logical_error_reward}, r_move={continue_reward}")
    plt.xlabel(r'$p_x$')
    plt.ylabel(r'Correct[\%] $p_s$')
    plt.legend()

    plt.savefig(plot_file_name)
    plt.show()

