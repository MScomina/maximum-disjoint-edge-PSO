import matplotlib.pyplot as plt
import numpy as np

execution_times_collated = {
    'Gurobi': [27.3711, 92.3655, 697.9784, 1024.0420, 1203.2094],
    'MSGA': [0.2577, 0.5105, 0.5555, 1.2402, 2.3249],
    'LaPSO': [32.9486, 42.1783, 70.2271, 125.1619, 267.2233]
}
std_devs_collated = {
    'Gurobi': [6.9248, 29.7768, 295.6846, 250.0163, 0.0114],
    'MSGA': [0.0215, 0.1561, 0.0536, 0.1617, 0.9221],
    'LaPSO': [2.2353, 5.2851, 14.3489, 19.6987, 66.9553]
}
instances_collated = [r'Collated$_{5,25}$', r'Collated$_{6,30}$', r'Collated$_{8,30}$', r'Collated$_{10,30}$', r'Collated$_{10,50}$']

execution_times_grid = {
    'Gurobi': [3.7978, 331.8981, 1200.2497, 1201.1051],
    'MSGA': [0.07477, 0.1690, 0.3348, 0.89112],
    'LaPSO': [30.0884, 67.3477, 92.4550, 201.6951]
}

std_devs_grid = {
    'Gurobi': [4.1768, 342.9462, 0.003563, 0.02573],
    'MSGA': [0.006059, 0.02746, 0.01926, 0.13363],
    'LaPSO': [3.0557, 15.2443, 16.9436, 12.2024]
}

instances_grid = [r'Grid$_{8,6}$', r'Grid$_{10,8}$', r'Grid$_{15,12}$', r'Grid$_{25,15}$']

max_time = 1200  # Maximum execution time

def plot_execution_times(times_dict, std_dict, instances, log_scale=False, show_std=True):
    plt.figure(figsize=(10, 6))
    for algorithm, times in times_dict.items():
        if show_std:
            std = std_dict[algorithm]
            lower_bounds = np.maximum(0.0001, np.array(times) - np.array(std))
            upper_bounds = np.minimum(max_time, np.array(times) + np.array(std))
            yerr = [np.abs(times - lower_bounds), np.abs(upper_bounds - times)]
            plt.errorbar(instances, times, yerr=yerr, marker='o', label=algorithm, capsize=5)
        else:
            plt.plot(instances, times, marker='o', label=algorithm)
    plt.axhline(y=max_time, color='r', linestyle='--', label=f'Max Time ({max_time}s)')
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Instances')
    plt.ylabel('Execution Time (s)')
    plt.title('Algorithm Execution Times')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.close()

def main():
    plot_execution_times(execution_times_collated, std_devs_collated, instances_collated)
    plot_execution_times(execution_times_collated, std_devs_collated, instances_collated, log_scale=True)
    plot_execution_times(execution_times_grid, std_devs_grid, instances_grid)
    plot_execution_times(execution_times_grid, std_devs_grid, instances_grid, log_scale=True, show_std=False)

if __name__ == '__main__':
    main()