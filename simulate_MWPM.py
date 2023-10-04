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


def simulate(simulation_settings):


    data_filename = 'data_' + simulation_settings['decoder'] + '.txt'



    sim_func = simulate_MWPM

    data = []
    all_px = gen_px_delta(simulation_settings['p_start'], simulation_settings['p_end'], simulation_settings['delta_p'])


    for L in simulation_settings['all_L']:
        print('L = ', L)
        for p_idx in range(len(all_px)):
            p = all_px[p_idx]
            if p_idx%3 == 0:
                print('progress: ', 100*round(p_idx/len(all_px),3), '%')
            k = 0
            for i in range(simulation_settings['N']):
                result = sim_func(L, p)
                if result:
                    k += 1
            if p==0.11:
                print(k)
            if p==0.167:
                print(k)
            data.append([L, p, k, simulation_settings['N']])


    if simulation_settings['save_data']:
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

    return data, all_data

def plot(plot_settings, data, all_data, success_rates, error_rates):

    
    plot_file_name = plot_settings['path'] + plot_settings['decoder'] + '_L=' + ','.join([str(x) for x in plot_settings['all_L']]) + '_N=' + str(
    plot_settings['N']) + 'p_start=' + str(plot_settings['p_start']).replace('.', ',') + 'p_end=' + str(plot_settings['p_end']).replace('.', ',') + '.pdf'
    all_L = plot_settings['all_L']

    if plot_settings['plot_all']:
        plot_data = all_data
    else:
        plot_data = data
    if plot_settings['tex_plot']:
        file_name = 'tex' + plot_file_name
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


    # plotting: 
    plt.figure()
    for i in range(len(all_L)):

        plt.errorbar(p_x[all_L[i]], p_corr[all_L[i]], yerr=errorbars[all_L[i]],linestyle='-.', label=r'$d = ' + str(all_L[i]) + '$' + " MWPM", linewidth = 0.5)

        plt.scatter(error_rates, success_rates*100, label=f"d={all_L[i]} PPO agent", color='darkblue', marker="^", s=30)
        #plt.scatter(np.linspace(0.05,0.25,5), success_rates_random*100, label=f"d={all_L[i]} PPO agent, trained & evaluated on random errors", color='darkblue', marker="^", s=30)
        #plt.scatter(np.linspace(0.05,0.25,5), success_rates_local*100, label=f"d={all_L[i]} PPO agent, trained & evaluated on local errors", color='red', marker="^", s=30)
        #plt.scatter(np.linspace(0.05,0.25,5), success_rates_random_local*100, label=f"d={all_L[i]} PPO agent, trained on random errors, evaluated on local errors", color='green', marker="^", s=30)
    plt.axis([plot_settings['p_start'], plot_settings['p_end'], 0, 100])
    #plt.title(r'Toric Code - ' + decoder + f" reward scheme r_succ={success_reward}, r_fail={logical_error_reward}, r_move={continue_reward}")
    plt.title(r'Toric Code - ' + plot_settings['decoder'])
    plt.xlabel(r'$p_x$')
    plt.ylabel(r'Correct[\%] $p_s$')
    plt.legend()

    plt.savefig(plot_settings['path'])
    plt.show()




