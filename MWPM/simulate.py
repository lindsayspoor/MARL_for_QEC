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

    # gridsizes to simulate:
    all_L = [3,5] #MWPM
    #all_L = [9, 17, 25, 33, 41]  # UF

    N = 1000  # number of simulations
    delta_p = 0.001  # distance between px # 0.00x = 0.x%
    p_start = 0.01
    p_end = 0.25
    plot_file_name = decoder + '_L=' + ','.join([str(x) for x in all_L]) + '_N=' + str(
        N) + 'p_start=' + str(p_start).replace('.', ',') + 'p_end=' + str(p_end).replace('.', ',') + '.png'
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



    success_rates_d_3 = np.loadtxt("success_rate_ppo_mlp_timesteps_1000000_lr_0.0005_d_3_sr_1.0_cr_0.0_ler_-1.0.csv", delimiter=',')
    success_rates_d_5 = np.loadtxt("success_rate_ppo_mlp_timesteps_1000000_lr_0.0005_d_5_sr_1.0_cr_0.0_ler_-1.0.csv", delimiter=',')
    plt.errorbar(p_x[3], p_corr[3], yerr=errorbars[3],linestyle='-.', label=r'$d = ' + str(3) + '$' + " MWPM", color='blue', linewidth = 0.5)
    plt.errorbar(p_x[5], p_corr[5], yerr=errorbars[5],linestyle='-.', label=r'$d = ' + str(5) + '$' + " MWPM", color='red', linewidth = 0.5)
    plt.scatter(np.linspace(0.05,0.25,5), success_rates_d_3*100, label="d=3 PPO agent", color='darkblue', marker="^", s=30)
    plt.scatter(np.linspace(0.05,0.25,5), success_rates_d_5*100, label="d=5 PPO agent", color='darkred', marker="^", s=30)
    plt.axis([p_start, p_end, 0, 100])
    plt.title(r'Toric Code - ' + decoder + r" reward scheme $r_{success}$=1, $r_{fail}$=-1, $r_{move}$=0")
    plt.xlabel(r'$p_x$')
    plt.ylabel(r'Correct[\%] $p_s$')
    plt.legend()

    plt.savefig(plot_file_name)
    plt.show()

