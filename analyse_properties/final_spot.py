"""
Processes properties files to compute mean values over last n frames within the polarity window and visualizes results.

Input:
directory: Directory containing property files.
average_over: Number of frames to average over.

Output:
Visualization of data properties through plots.
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.label_and_analyse_plotting import plot_single_parameter

# Use the darkgrid theme for seaborn
sns.set_theme(style="darkgrid")


def concatenate_and_fill_na(df1, df2):
    # Concatenate the DataFrames along the columns axis
    result = pd.concat([df1, df2], ignore_index=True)

    # Fill missing values with appropriate values
    result = result.ffill().bfill()

    return result


def calculate_mean_values(df, average_over):
    """
    Calculate the mean of numeric values in DataFrame over a certain number of rows,
    keeping non-numeric values consistent across all rows.

    Parameters:
        df (DataFrame): Input DataFrame.
        average_over (int): Number of rows to average over.

    Returns:
        DataFrame: A DataFrame containing the mean values.
    """

    # Check if DataFrame is not empty
    if not df.empty:
        # Get the values of the last rows
        last_rows = df.tail(average_over)

        # Calculate the mean of numeric columns
        mean_values = last_rows.select_dtypes(include='number').mean()

        # Get the last row for non-numeric columns
        last_non_numeric_values = last_rows.select_dtypes(exclude='number').iloc[-1]

        # Ensure both mean_values and last_non_numeric_values are single-row DataFrames
        mean_values = mean_values.to_frame().T
        last_non_numeric_values = last_non_numeric_values.to_frame().T
        # Concatenate mean numeric values with last non-numeric values
        mean_df = concatenate_and_fill_na(mean_values, last_non_numeric_values)

        return mean_df

    else:
        # Return None if DataFrame is empty
        return None


def remove_last_folder(directory):
    # Find the index of the last occurrence of '/'
    last_slash_index = directory.rfind('/')

    # If '/' is found, remove everything after it
    if last_slash_index != -1:
        return directory[:last_slash_index + 1]
    else:
        # If '/' is not found, return the original directory
        return directory


# Input
location = os.getcwd()
directory = remove_last_folder(location)
folder = "properties/1_spot/"
directory = directory + '/' + folder

directory = ''
average_over = 3

# Start of code
direc = os.listdir(directory)
dataframes = []

# Find properties at end of polarity window
# Iterate through each file in the directory
for file in os.listdir(directory):

    if file.endswith('.csv'):
        # Read CSV file into a DataFrame
        df = pd.read_csv(os.path.join(directory, file))

        # Filter DataFrame to get only part in polarity window
        df = df[df['in_window'] == True]

        # Compute mean values over last n frames
        if len(df) > average_over:
            mean_values = calculate_mean_values(df, average_over)
            if mean_values is not None:
                dataframes.append(mean_values.head(1))


# Construct a new DataFrame from the list of last rows
if dataframes:
    result_df = pd.concat(dataframes)
    result_df.reset_index(drop=True, inplace=True)  # Reset the index
else:
    print("No data available.")

hue_order = []
result_df['strain'] = pd.Categorical(result_df['strain'])

# Set the categories based on hue_order
result_df['strain'] = result_df['strain'].cat.set_categories(hue_order)

# Sort the DataFrame based on the 'strains' column
result_df = result_df.sort_values(by='strain')

plot_single_parameter(result_df, 'surface', hue_order=[''], title='Surface  polarity spot', xlabel=None, ylabel='')

# Define the single combination to plot
param_combination = ['volume', 'total_int']

# Compute the correlation between the selected parameters
correlation = result_df[param_combination[0]].corr(result_df[param_combination[1]])

# Create a scatter plot for the selected combination
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(x=param_combination[0], y=param_combination[1], data=result_df, hue='strain', palette='colorblind', alpha=0.5)

# Set x-axis and y-axis labels
scatter_plot.set_xlabel('Volume [µm³]')
scatter_plot.set_ylabel('Total intensity (relative)')

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.text(0.5, 0.95, f'Correlation: {correlation:.2f}', horizontalalignment='left', verticalalignment='top', transform=scatter_plot.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Set the title for the plot
plt.title('')

# Display the plot
plt.show()
