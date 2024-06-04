import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Use the darkgrid theme for seaborn
sns.set_theme(style="darkgrid")

"""
info
"""

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
        return df  # Return the original DataFrame instead of False

    relative_time = list(range(-1 * switch_index + 1, len(df_reset) - switch_index + 1))
    df_reset['relative_time'] = relative_time  # Remove the comma here
    return df_reset  # Return the modified DataFrame
def parse_centroid(centroid_str):
    centroid_str = centroid_str.strip('()')  # Remove parentheses
    coordinates = centroid_str.split(',')  # Split by comma
    return tuple(float(coord) for coord in coordinates)

def cooc_start_end(df):
    # Initialize variables
    start_index = None
    end_index = None

    for index, row in df.iterrows():
        # If current row value is 2 and start_index is not set, set start_index
        if row['n_spots'] == 2 and start_index is None:
            start_index = row['time']
        # If start_index is set and current row value is 2, update end_index
        elif start_index is not None and row['n_spots'] == 2:
            end_index = row['time']
    return start_index,end_index

def cooc_start_end_rel(df):
    # Initialize variables
    start_index = None
    end_index = None

    for index, row in df.iterrows():
        # If current row value is 2 and start_index is not set, set start_index
        if row['n_spots'] == 2 and start_index is None:
            start_index = row['relative_time']
        # If start_index is set and current row value is 2, update end_index
        elif start_index is not None and row['n_spots'] == 2:
            end_index = row['relative_time']
    return start_index,end_index


def in_window(df):
    # Extract first and last rows
    first_row = bool(df.iloc[0]['in_window'])
    last_row = bool(df.iloc[-1]['in_window'])
    print(last_row)

    # Check if first and last rows have the same values
    if  last_row is True and first_row is True:
        return "in"
    elif last_row is False and first_row is False:
        return 'out'
    else:
        return 'partially' # note this means the coocurence is at the edge

directory = ''
direc = os.listdir(directory)

properties = pd.DataFrame(columns= ['strain','time','window'])

#############
# Set consistent style for seaborn
sns.set_context("talk", font_scale=1.1)
sns.set_style("whitegrid")

# Set consistent style for matplotlib
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

filtered_dataframes = []

for file in os.listdir(directory):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, file))

        # Find the index of the first occurrence of True
        first_true_idx = df['in_window'].idxmin() if df['in_window'].any() else None

        # If there is a True value, set all rows before and including that index to True
        if first_true_idx is not None:
            df.loc[:first_true_idx, 'in_window'] = True

        if len(df) >= 5:
            # Keep only part dataframe where there is co-occurrence
            start, end = cooc_start_end(df)
            filtered_df = df[(df['time'] >= start) & (df['time'] <= end)]
            filtered_dataframes.append(filtered_df)

# Combine all filtered dataframes
final_df = pd.concat(filtered_dataframes)


# Plot the co-occurrence time without hue
sns.violinplot(data=final_df, y='time', x='strain', palette='Set3')
sns.boxplot(data=final_df, y='time', x='strain', dodge=True, linewidth=1.3,
            flierprops=dict(marker='x', markersize=5), fill=None,
            palette='Set2')

# Annotate strain names with number of files
num_files_per_strain = final_df.groupby('strain').size().reset_index(name='num_files')
for i, row in num_files_per_strain.iterrows():
    plt.text(i, 0, f"(N={row['num_files']})", ha='center')


plt.ylabel('Co-occurrence [s]')
plt.title('Co-occurrence Time')
plt.tight_layout()
plt.show()



############
#
# for file in direc:
#     df = pd.read_csv(directory + file)
#
#     # Find the index of the first occurrence of True
#     first_true_idx = df['in_window'].idxmin() if df['in_window'].any() else None
#     # If there is a True value, set all rows before and including that index to True
#     if first_true_idx is not None:
#         df.loc[:first_true_idx, 'in_window'] = True
#
#
#     if len(df) >= 5:
#         # Keep only part dataframe where there is co-occurrence
#         start, end = cooc_start_end(df)
#         filtered_df = df[(df['time'] >= start) & (df['time'] <= end)]
#
#
# print(properties)
# # plot window
#
# # timing of spots
# fig, ax= plt.subplots(2)
# fig.suptitle('Timing of coocuring spots')
#
# ax[0].set_title('In or after polarity window')
# sns.countplot(data=properties, x='strain', stat="percent", hue='window', ax=ax[0])
#
# ax[1].set_title('Coocurence time [frames]')
# # sns.violinplot(data=properties, y='time',x='strain', ax=ax[1], cut=0)
# sns.swarmplot(data=properties, y='time',x='strain', ax=ax[1])
#
# plt.show()


##########

cooc_df = pd.DataFrame()

for file in direc:
    df = pd.read_csv(directory + file)

    if df.empty:
        continue

    # Ensure 'in_window' is a boolean column and contains at least one True value
    if not df['in_window'].any():
        print(f"No True values found in 'in_window' column for file {file}.")
        continue

    first_true_index = df[df['in_window']].index[0]
    # Set all rows before the first True value to True
    df.loc[:first_true_index, 'in_window'] = True
    # Reset the index if needed
    df.reset_index(drop=True, inplace=True)

    # Update the DataFrame with relative time information
    df = time_relative_to_window(df)
    if 'relative_time' in df.columns:
        print(df['relative_time'])
    else:
        print(f"Failed to add 'relative_time' for file {file}.")
    if not df['relative_time'].empty:
        start, end = cooc_start_end_rel(df)
        # Add rows with co-occurring spots to the cooc_df DataFrame
        cooc_df = cooc_df._append(df.loc[start:start])

# Plot histogram of relative time points for co-occurring spots
plt.hist(cooc_df['relative_time'], bins=10, edgecolor='black')
plt.title('Histogram of Relative Time Points for Co-occurring Spots')
plt.xlabel('Relative Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

g = sns.FacetGrid(cooc_df, col='strain', col_wrap=3, margin_titles=True)
g.map(plt.hist, 'relative_time', bins=10, edgecolor='black')
g.set_axis_labels('Relative Time', 'Frequency')
g.set_titles(col_template="{col_name}")
g.fig.suptitle('Histogram of Relative Time Points for Co-occurring Spots per Strain', y=1.05)
plt.show()
