import os
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from lib.label_and_analyse_plotting import (plot_strains_in_separate_figures, plot_strains_in_one_figure)
sns.set_theme(style="darkgrid")

def time_relative_to_window(df):
    if df.empty:
        return df

    df_reset = df.reset_index(drop=True)
    diff_values = df_reset['in_window'].diff()

    switch_index = None
    for i, value in enumerate(diff_values):
        if value is True:
            switch_index = i
            break

    if switch_index is None:
        print("No switch indices found.")
        return False

    relative_time = list(range(-1 * switch_index + 1, len(df_reset) - switch_index + 1))
    df_reset['relative_time'] = relative_time

    return df_reset

def values_over_time(df, values_df):
    if df.empty:
        return df

    if values_df is None:
        values_df = pd.DataFrame(columns=df.columns)
        values_df['num'] = pd.Series(dtype='int')
        for column in df.columns:
            if column != 'strain' and column != 'relative_time' and df[column].dtype != 'object':
                values_df[column + '_values'] = [[] for _ in range(len(values_df))]

    strain = df.at[0, 'strain']

    values_df_reset = values_df.reset_index(drop=True)

    for index, row in df.iterrows():
        rel_time = int(row['relative_time'])

        desired_row = values_df_reset[(values_df_reset['strain'] == strain) & (values_df_reset['relative_time'] == rel_time)]

        if not desired_row.empty:
            num = desired_row['num'].values[0]

            for column in df.columns:
                if column != 'strain' and column != 'relative_time' and column != 'in_window' and df[column].dtype != 'object':
                    values_list = desired_row[column + '_values'].values[0]
                    value = row[column]
                    values_list.append(value)
                    values_df.at[desired_row.index[0], column + '_values'] = values_list

            num += 1
            values_df.at[desired_row.index[0], 'num'] = num

        else:
            mean_to_add = row.copy()
            mean_to_add['num'] = 1
            for column in df.columns:
                if column != 'strain' and column != 'relative_time' and column != 'in_window' and df[column].dtype != 'object':
                    mean_to_add[column + '_values'] = [row[column]]
            values_df = values_df._append(mean_to_add, ignore_index=True)

    return values_df


def compute_mean_std_col(df, column_name):
    mean_values = []
    std_values = []
    num_samples = []  # Store number of samples for each row
    removed_files = []  # Store files removed

    for index, row in df.iterrows():
        data = row[column_name]
        if not data:  # Check if the list is empty
            mean_values.append(np.nan)
            std_values.append(np.nan)
            num_samples.append(0)  # Set number of samples to 0
            continue  # Skip further processing for this row

        mean = statistics.mean(data)  # Compute mean
        std = np.std(data)  # Compute standard deviation

        # Remove outliers that are 3 standard deviations away from the mean
        filtered_data = data #[x for x in data if (x >= mean - 3 * std) and (x <= mean + 3 * std)]
        if len(data) != len(filtered_data):
            removed_files.append(index)

        mean_values.append(statistics.mean(filtered_data))
        std_values.append(np.std(filtered_data))
        num_samples.append(len(filtered_data))  # Store number of samples

    df[column_name + '_mean'] = mean_values
    df[column_name + '_std'] = std_values
    df['Number_of_samples'] = num_samples  # Add number of samples column

    return df, removed_files

def calculate_mean_std(df, columns=None):
    """
    Calculate mean and standard deviation for lists of values in specified columns of the DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing columns with lists of values.
        columns (list or None): List of column names containing lists of values for which mean and std will be calculated.
            If None, automatically identify columns containing lists of values.

    Returns:
        pandas.DataFrame: DataFrame with added columns for mean and standard deviation.
    """
    if columns is None:
        columns = [col for col in df.columns if isinstance(df[col][0], list)]

    for column in columns:
        new_df, removed = compute_mean_std_col(df, column)

        # print(removed)

    return new_df


def remove_lowN(mean_df, percentage):
    # Get unique strains
    unique_strains = mean_df['strain'].unique()

    # Create an empty list to store the filtered DataFrames for each strain
    filtered_list = []

    # Loop over each unique strain
    for strain in unique_strains:
        # Filter the DataFrame to get data for only the current strain
        strain_df = mean_df[mean_df['strain'] == strain]

        # Get the maximum number of samples for this strain
        max_samples = strain_df['Number_of_samples'].max()

        # Calculate the threshold as a percentage of the maximum number of samples
        threshold = (percentage / 100) * max_samples

        # Filter out rows that have fewer than the threshold number of samples
        filtered_strain_df = strain_df[strain_df['Number_of_samples'] >= threshold]

        # Append the filtered data for this strain to the list
        filtered_list.append(filtered_strain_df)

    # Concatenate all filtered DataFrames into one
    filtered_df = pd.concat(filtered_list, ignore_index=True)

    return filtered_df


directory = ''
direc = os.listdir(directory)

frametime = 0

# define already
values_df = None

correlations = pd.DataFrame(columns=['strain', 'vol_intx', 'vol_surf', 'surf_int', 'intint'])

for file in direc:
    df = pd.read_csv(directory + file)
    # remove colums that are not needed for this code
    df_fil = df.filter(['strain','volume','surface', 'total_int', 'intensity_max', 'in_window', 'bbox', 'time']) # keep only these columns

    # add filename for plotting all
    df_fil['file'] = file

    # remove time before first True (otherwise will have 0 at this flip)
        # Find the index of the first True value in the 'in_window' column
    first_true_index = df_fil[df_fil['in_window']].index[0]
        # Select rows starting from the first True value
    df = df_fil.loc[first_true_index:]
        # If you want to reset the index
    df.reset_index(drop=True, inplace=True)

    # add relative time column
    df = time_relative_to_window(df)
    if df is False:
        continue # skip this one

    values_df = values_over_time(df, values_df)

# compute means and stds
mean_df = calculate_mean_std(values_df)

# only keep time points where there is a certein percentage of samples
mean_df = remove_lowN(mean_df, 30)

# rescale time
mean_df['relative_time'] = mean_df['relative_time'].multiply(frametime)

# Define sigmoid function
def sigmoid(x, k, x0):
    x = np.array(x, dtype=float)  # Convert input to NumPy array with float dtype
    return 1 / (1 + np.exp(-k * (x - x0)))

column_headers = list(mean_df.columns.values)
# print(column_headers)
mean_df = mean_df.drop(columns=['time'])
mean_df.rename(columns={'relative_time': 'time', 'num': 'n_samples'}, inplace=True)

from scipy.optimize import curve_fit
def exponential_fit_function(x, A, B):
    """Exponential fitting function going through the origin."""
    return A * np.exp(B * x)

def sigmoid(x,a,b,c):
    return a/(b + np.exp(-c*x))

def linear_fit_function(x, m, c):
    return m * x + c

    # Function to fit linear line and print slope
def fit_linear_and_print_slope(x, y, start_time, end_time):
    mask = (x >= start_time) & (x <= end_time)
    x_window = x[mask]
    y_window = y[mask]
    popt, _ = curve_fit(linear_fit_function, x_window, y_window)
    slope = popt[0]
    print("Slope:", slope)


# sort based on time
mean_df = mean_df.sort_values(by='time')

params = []

# Loop through each parameter
for param in params:
    # Loop through each strain
    y = param + '_values_mean'

    if 'std' in mean_df.columns:
        mean_df = mean_df.drop(columns=['std'])
    mean_df.rename(columns={param + '_values_std': 'std'}, inplace=True)


    plot_strains_in_separate_figures(data=mean_df, x='time', y=y, strains=[ ],
                     palette='Set2',
                     fit_func='',fit_params=(-100,0,15,30), slope_unit = '', # length is in frames not minutes
                     show_line=True, show_conf=True,
                     title='', xlab='', ylab='',
                                    )
    plt.show()

