import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Use the darkgrid theme for seaborn
sns.set_theme(style="darkgrid")


def plot_parameter(df, param, title):
    if len(df) >= 10:
        fig, ax = plt.subplots()

        sns.lineplot(ax=ax, x='time', y=param, data=df, palette='Set1', marker='', color='k')
        sns.lineplot(ax=ax, x='time', y=param, data=df, palette='Set1', marker='', color='k')
        sns.lineplot(ax=ax, x='time', y=param, hue='phase', data=df, palette='Set1', marker='o', linestyle='')

        # Shade the entire height where 'in_window' is True
        ax.fill_between(df['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=df['in_window'], color='blue', alpha=0.1)



def plot_parameters(df, params, title):
    if len(df) >= 10:
        num_params = len(params)
        num_cols = min(num_params, 2)
        num_rows = (num_params + 1) // 2

        fig, axes = plt.subplots(num_rows, num_cols)
        # axes = axes.flatten()

        for i, param in enumerate(params):
            sns.lineplot(ax=axes[i], x='time', y=param, data=df, palette='Set1', marker='', color='k')
            sns.lineplot(ax=axes[i], x='time', y=param, data=df, palette='Set1', marker='', color='k')
            sns.lineplot(ax=axes[i], x='time', y=param, hue='in_window', data=df, palette='Set1', marker='o', linestyle='')
            # Shade the entire height where 'in_window' is True
            axes[i].fill_between(df['time'], axes[i].get_ylim()[0], axes[i].get_ylim()[1], where=df['in_window'], color='blue', alpha=0.1)


"""
Visualization of parameters over time per file

Input:
directory: Directory containing property files.
file: location of a single file you want to run
run_direc: if True runs code for entire directory, if False only for the specified file
params: parameters which it will plot

Output:
Visualization of data properties through plots.
"""

# decide if you want to run a directory or look at a single file
run_direc = True

# input file locations
directory = ''
file = directory + ""

# decide which parameters you want to see
params = [
]

# main code
if run_direc:
    direc = os.listdir(directory)
    for file in direc:

        print(file)

        df = pd.read_csv(os.path.join(directory, file))

        # identify phase based on in_window
        last_true_index = df.index[df['in_window']].max() if any(df['in_window']) else -1
        first_true_index = df.index[df['in_window']].min() if any(df['in_window']) else len(df)

        df['phase'] = 'establishment'
        df.loc[last_true_index + 1:, 'phase'] = 'maintenance'
        df.loc[:first_true_index - 1, 'phase'] = 'before establishment'

        df['time'] = (1/3) * df['time']

        plot_parameter(df, 'volume', file)
        # plot_parameters(df, params, file)

        plt.title('Volume over time')
        plt.xlabel('Time [min]')
        plt.ylabel('Volume [µm³]')

        plt.show()


else:
    df = pd.read_csv(file)
    plot_parameters(df, params, file)
