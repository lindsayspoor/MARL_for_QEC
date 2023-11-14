import numpy as np
from simulate_MWPM import simulate, plot_threshold

p_start = 0.01 
p_end = 0.20
lambda_value=1

simulation_settings = {'decoder': 'MWPM',
                    'N': 1000,
                    'delta_p': 0.001,
                    'p_start': p_start,
                    'p_end': p_end,
                    'path': f'Figure_results/Results_benchmarks/MWPM_threshold_plot_local_errors.pdf',
                    'tex_plot' : False,
                    'save_data' : True,
                    'plot_all' : True,
                    'all_L':[5],
                    'random_errors':True,
                    'lambda_value':lambda_value}

plot_settings = simulation_settings
plot_settings['all_L']=[5]

#sim_all_data = np.loadtxt("data_MWPM_random_True.txt")

sim_data, sim_all_data = simulate(simulation_settings)

plot_threshold(plot_settings, sim_all_data)