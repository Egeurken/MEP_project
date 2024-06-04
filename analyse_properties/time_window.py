import os
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from lib.label_and_analyse_plotting import plot_all_strains

def time_relative_to_window(df):
    if df.empty:
        # If the DataFrame is empty, return it as is
        return df

    # Reset index before performing comparison
    df_reset = df.reset_index(drop=True)

    # Find the index where 'in_window' switches from True to False
    diff_values = df_reset['in_window'].diff()

    # Find the index of the first occurrence of True
    switch_index = None
    for i, value in enumerate(diff_values):
        if value is True:
            switch_index = i
            break

    if switch_index == None:
        # Add debugging output to understand the issue
        print("No switch indices found.")
        # print(diff_values)
        return False

    # Calculate relative_time
    relative_time = list(range(-1 * switch_index + 1, len(df_reset) - switch_index + 1))

    # Add the 'relative_time' column to the DataFrame
    df_reset['relative_time'] = relative_time

    return df_reset

def values_over_time(df, mean_df):
    if df.empty:
        return df

    if mean_df is None:
        mean_df = pd.DataFrame(columns=df.columns)
        mean_df['num'] = pd.Series(dtype='int')
        for column in df.columns:
            if column != 'strain' and column != 'relative_time' and df[column].dtype != 'object':
                mean_df[column + '_values'] = [[] for _ in range(len(mean_df))]

    strain = df.at[0, 'strain']

    mean_df_reset = mean_df.reset_index(drop=True)

    for index, row in df.iterrows():
        rel_time = int(row['relative_time'])

        desired_row = mean_df_reset[(mean_df_reset['strain'] == strain) & (mean_df_reset['relative_time'] == rel_time)]

        if not desired_row.empty:
            num = desired_row['num'].values[0]

            for column in df.columns:
                if column != 'strain' and column != 'relative_time' and column != 'in_window' and df[column].dtype != 'object':
                    values_list = desired_row[column + '_values'].values[0]
                    value = row[column]
                    values_list.append(value)
                    mean_df.at[desired_row.index[0], column + '_values'] = values_list

            num += 1
            mean_df.at[desired_row.index[0], 'num'] = num

        else:
            mean_to_add = row.copy()
            mean_to_add['num'] = 1
            for column in df.columns:
                if column != 'strain' and column != 'relative_time' and column != 'in_window' and df[column].dtype != 'object':
                    mean_to_add[column + '_values'] = [row[column]]
            mean_df = mean_df._append(mean_to_add, ignore_index=True)

    return mean_df


def compute_mean_std_col(df, column_name):
    # Initialize empty lists to store mean and std values
    mean_values = []
    std_values = []

    # Loop over each row in the specified column
    for row in df[column_name]:
        # Compute mean and std for the list in each row
        mean = statistics.median(row)
        std = np.std(row)

        # Append mean and std values to respective lists
        mean_values.append(mean)
        std_values.append(std)

    # Add mean and std columns to the dataframe
    df[column_name + '_mean'] = mean_values
    df[column_name + '_std'] = std_values

    return df

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
    new_df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original DataFrame

    # Identify columns containing lists of values if not provided
    if columns is None:
        columns = [col for col in df.columns if isinstance(df[col][0], list)]
        # print(columns)

    for column in columns:
        new_df = compute_mean_std_col(new_df, column)

    return new_df

def calculate_bbox_param(bbox):
    return  (bbox[2]**2) / (bbox[0]*bbox[1])

def heigth(bbox):
    # return  (bbox[2]**2) / (bbox[0]*bbox[1])
    return  bbox[2]

def area(bbox):
    # return  (bbox[2]**2) / (bbox[0]*bbox[1])
    return  bbox[0]* bbox[1]


"""
info
"""

directory = ''

direc = os.listdir(directory)

frametime = 1/3

# define already
values_df = None

correlations = pd.DataFrame(columns=['strain', 'vol_intx', 'vol_surf', 'surf_int', 'intint'])

for file in direc:
    df = pd.read_csv(directory + file)
    # remove colums that are not needed for this code
    df_fil = df.filter(['strain','volume','surface', 'total_int', 'intensity_max', 'in_window', 'bbox']) # keep only these columns

    print(file)

    # remove time before first True (otherwise will have 0 at this flip)
        # Find the index of the first True value in the 'in_window' column
    first_true_index = df_fil[df_fil['in_window']].index[0]
        # Select rows starting from the first True value
    df = df_fil.loc[first_true_index:]
        # If you want to reset the index
    df.reset_index(drop=True, inplace=True)

    # bbox stuff
    df['bbox'] = df['bbox'].apply(ast.literal_eval)

    df['heigth'] = df['bbox'].apply(lambda x: heigth(x))
    df['area'] = df['bbox'].apply(lambda x: area(x))

    # add relative time column
    df = time_relative_to_window(df)
    if df is False:
        continue # skip this one

    values_df = values_over_time(df, values_df)

# compute means and stds
mean_df = calculate_mean_std(values_df)

# print(mean_df.to_string())
print(mean_df.head(10))
    # values over time
    # mean_df = values_over_time(df,mean_df)


# only keep time points where there are 10 samples
mean_df = mean_df[mean_df['num'] >= 15]
mean_df = mean_df[mean_df['strain'] != 'axl2Δ rax1Δ ']

# rescale time
mean_df['relative_time'] = mean_df['relative_time'].multiply(frametime)

# plot
pd.set_option('display.max_columns', None)
# Print the first 5 rows of the DataFrame
print(mean_df.head())


numeric_parameters = ['volume']

# Calculate the number of rows required based on the number of numeric parameters
num_rows = (len(numeric_parameters) + 1) // 2

# Create a figure and axis objects using subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

######## plot
for param in numeric_parameters:
    plot_all_strains(mean_df,x='relative_time', y=param + '_values_mean',show_line=True, show_fit=False, show_std=True, separate_plots=True)

