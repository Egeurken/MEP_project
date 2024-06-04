import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import vg

from matplotlib.patches import Patch, FancyArrowPatch
from scipy import stats

from lib.label_and_analyse_plotting import (plot_strains_in_separate_figures)
"""  
Uses the weighted center of mass from the properties file to compute the stepsizes and MSD

Input:
directory: Directory containing property files.
max_stepsize: determines what the maximum stepsize is to be included in analysis (and which as seen as outliers)
MSD_inwindow: determines is you make the MSD plot for full window or only part (for False stepsize plot gets affected)
MSD_minN : denotes how many samples must be present for a timepoint to be included in MSD plot

plot: denotes for which indexes a 3D plot of the path will be made
scale: scale of imaging (coordinates where still in voxels, will change in next version)

Output:
stepsize plot, sepated per wether in window
MSD plot, showing MSD, standerd deviation and the slope of the curve
"""

########################### Functions

def parse_centroid(centroid_str):
    """
    Parses a string representing a centroid into a tuple of floating-point numbers.

    Parameters:
        centroid_str (str): A string representing a centroid in the format '(x, y, z)'.

    Returns:
        tuple: A tuple containing the x, y, and z coordinates of the centroid as floats.
    """
    centroid_str = centroid_str.strip('()')  # Remove parentheses
    coordinates = centroid_str.split(',')  # Split by comma
    return tuple(float(coord) for coord in coordinates)


def distance_3d(point1, point2):
    """
    Calculates the Euclidean distance between two points in 3D space.

    Parameters:
        point1 (tuple): A tuple containing the x, y, and z coordinates of the first point.
        point2 (tuple): A tuple containing the x, y, and z coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return abs(np.sqrt(((point2[0] - point1[0])) ** 2 + ((point2[1] - point1[1])) ** 2 + ((point2[2] - point1[2])) ** 2))


def angle(p1, p2, acute):
    """
    Calculates the angle between two vectors.

    Parameters:
        p1 (tuple): First position.
        p2 (tuple): Second position.
        acute (bool): If True, returns the acute angle, else returns the actual angle.

    Returns:
        float: The angle between the two vectors.
    """
    # Convert tuples to NumPy arrays for easy calculation
    p1 = np.array(p1)
    p2 = np.array(p2)

    vec1 = p2 - p1
    vec2 = np.array([1, 0, 0])

    # angle = vg.angle(vec1, vec2, look=vg.basis.z)
    # angle = vg.signed_angle(vec1, vec2, look=vg.basis.z,  units='deg')
    angle = vg.angle(vec1, vec2, look=vg.basis.z, units='deg')
    print(angle)

    if not acute:
        # Adjust angle to be between 0 and 2*pi
        angle = angle if angle <= np.pi else 2 * np.pi - angle

    return angle


def distance_previous_frame(df, scale):
    """
    Calculates the scaled Euclidean distances between consecutive frames in a DataFrame.

    Parameters:
        df (DataFrame): properties dataframe centroid information for each frame of a file.
        scale (float): used to convert pixel units to um

    Returns:
        list: A list containing the scaled Euclidean distances between consecutive frames.
    """
    if len(df) < 2:
        return []  # Empty list if there is only one frame

    distances_prev = [0]  # First row has distance 0
    # angle_prev = [0]

    prev_position = parse_centroid(df.iloc[0]['centroid'])  # Initial position

    for i in range(1, len(df)):
        current_position = parse_centroid(df.iloc[i]['centroid'])

        # determine the distance traveled
        distance = scale * distance_3d(prev_position, current_position)
        distances_prev.append(distance)

        # # determine angle of movement
        # angles = angle(current_position, prev_position, True)
        # angle_prev.append(angles)

        prev_position = current_position

    return distances_prev


def plot_path_with_actual(df, file):
    """
    Plots the reconstructed path based on distances and angles, as well as the actual path from the 'centroid' column
    in the DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the actual path in the 'centroid' column.
        file (str): File name for the plot title.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Actual path
    x_actual = [parse_centroid(coord)[0] for coord in df['centroid']]
    y_actual = [parse_centroid(coord)[1] for coord in df['centroid']]
    z_actual = [parse_centroid(coord)[2] for coord in df['centroid']]

    # Interpolate colors
    num_points = len(x_actual)
    colors = plt.cm.jet(np.linspace(0, 1, num_points))

    for i in range(num_points - 1):
        ax.plot(x_actual[i:i+2], y_actual[i:i+2], z_actual[i:i+2], color=colors[i], label='Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Path of: {file[6:-4]}')

    plt.show()


def add_new_column(df, x):
    df['new_column'] = False
    true_indices = df.index[df['in_window'] == True][:x]
    df.loc[true_indices, 'T1'] = True


########################### Input
directory = ''
direc = os.listdir(directory)

max_stepsize = 0  # um, denotes the threshold of stepsizes
MSD_inwindow = None  # True/False: only computes ]msd for where 'in_window' is t/f, in None runs for full window
MSD_minN = 0

plot = []  # indexes of paths to plot

scale = 0
########################### code

# Define output storage
movement_df = pd.DataFrame()

# Set plotting style
sns.set_theme(style="darkgrid")

# Loop over all files in directory
for i, file in enumerate(direc):
    # Load a single parameter file
    df = pd.read_csv(directory + file)

    if MSD_inwindow is False:
        df = df[df['in_window'] == False].reset_index(drop=True)

    if 3 <= len(df) < 100:  # Only take relevant time windows (filter out nonsense data)

        # Compute distance with previous frame
        distances_prev = distance_previous_frame(df, scale)

        # Assign distances to 'distance_from_previous' column
        df['stepsize'] = distances_prev

        # Remove sections containing large jumps
        for index, row in df.iterrows():
            if row['stepsize'] > max_stepsize:
                if row['in_window']:
                    df = df.iloc[:index, :]  # Remove the current row and all rows before it
                else:
                    df = df.iloc[index + 1:, :]  # Remove the current row and all rows after it

        if len(df) < 5:
            continue

        # Keep only T1 estimated
        true_count = 0

        # Iterate through the DataFrame and update values accordingly
        for index, row in df.iterrows():
            if row['in_window']:
                true_count += 1
                if true_count > 9:
                    df.at[index, 'in_window'] = False

        # Create 'time' column containing sequential numbers starting from 0
        time = list(range(len(df)))

        # Add time column to dataframe and convert to minutes (instead of frames)
        df['time'] = [(1 / 3) * t for t in time]

        # Add file indicator column
        df['file'] = file

        if i in plot:
            try:
                plot_path_with_actual(df, file)
            except KeyError as err:
                print(f"KeyError occurred: {err}")

        # Compute displacement for each time point
        displacement = []
        start_pos = parse_centroid(df.iloc[0]['centroid'])  # Initial position
        for _, row in df.iterrows():
            current_position = parse_centroid(row['centroid'])
            distance = scale * distance_3d(start_pos, current_position)
            displacement.append(abs(distance))

        # Get squared displacement
        SD = [x ** 2 for x in displacement]

        # Add SD to dataframe
        df['SD'] = SD

        # Add results to movement_df
        df_fil = df.filter(['strain', 'time', 'in_window', 'stepsize', 'SD', 'file'])  # These are the only needed columns
        movement_df = pd.concat([movement_df, df_fil])

# preproces data for stepsizes
    # remove rows where time = 0
filtered_df = movement_df[movement_df['time'] != 0]

####################### plot stepsizes

# can uncomment this to look at what the values are of stepsizes witout avaraging over a file
#
# hue_order = ['True', 'False']
# sns.boxplot(data=filtered_df, y='stepsize', x='strain',hue='in_window',
#             palette = "Set2")
# plt.ylabel('Stepsize [um]')
# plt.title('Displacement per 20 seconds')
# plt.show()

########### plot stepsizes over time
# Grouping by 'strain' and 'time', and calculating the mean of 'stepsize' for each group
filtered_df1 = filtered_df.groupby(['strain', 'time']).filter(lambda x: len(x) > 5)

# Recalculate mean stepsize after filtering
new_df = filtered_df1.groupby(['strain', 'time']).agg({'stepsize': 'mean'}).reset_index()

# Renaming the columns for clarity
new_df.columns = ['strain', 'time', 'stepsize']

# Plotting using Seaborn
sns.lineplot(data=new_df, x='time', y='stepsize', hue='strain', palette='Set2')
plt.ylim(bottom=0)

# Adding title and labels
plt.title('Mean Stepsize Over Time')
plt.xlabel('Time')
plt.ylabel('Mean Stepsize')

# Displaying the plot
plt.show()

########### plot avaraged stepsizes
def bootstrap_median(data, num_bootstrap_samples=1000):
    medians = np.empty(num_bootstrap_samples)
    n = len(data)

    for i in range(num_bootstrap_samples):
        sample = np.random.choice(data, size=n, replace=True)
        medians[i] = np.median(sample)

    return np.std(medians, ddof=1)


# Calculate number of different files per strain
num_files_per_strain = filtered_df.groupby('strain')['file'].nunique().reset_index(name='num_files')

# Group by 'strain', 'file', and 'in_window'
grouped = filtered_df.groupby(['strain', 'file', 'in_window'])

# Calculate the mean stepsize for each file
averaged_SS = grouped['stepsize'].mean().reset_index()

# Create a new column 'window' based on 'in_window'
averaged_SS['phase'] = averaged_SS['in_window'].apply(lambda x: 'T1' if x else 'T2')

# include the average of the full file
full_file_avg = filtered_df.groupby(['strain', 'file'])['stepsize'].mean().reset_index()
full_file_avg['phase'] = 'full file'

# Concatenate the dataframes
final_df = pd.concat([averaged_SS, full_file_avg], ignore_index=True)
final_df = averaged_SS

# Define the order of strains
strains = ['WT', 'rsr1Δ', 'axl2Δ rax1Δ']

# Merge num_files_per_strain with final_df
final_df = pd.merge(final_df, num_files_per_strain, on='strain', how='left')

# Drop the 'in_window' column if not needed anymore
final_df = final_df.drop(columns=['in_window'])

# Define hue order
hue_order = ['T1', 'T2']

# Define the color for the box plot
boxplot_color = 'dimgray'

# Create the violin plot
sns.violinplot(data=final_df, y='stepsize', x='strain', hue='phase', hue_order=hue_order, palette='Set3', order=strains)

# Add box plot with only the outline and specify the color
sns.boxplot(data=final_df, y='stepsize', x='strain', hue='phase', hue_order=hue_order, dodge='auto',
            linewidth=1.3, flierprops=dict(marker='x', markersize=5), fill=None, legend=False,
            gap=0.6, order=strains,
            palette={window: boxplot_color for window in hue_order})

# Annotate strain names with number of files
for i, row in num_files_per_strain.iterrows():
    plt.text(i, 0, f"(N={row['num_files']})", ha='center')


plt.ylabel('Stepsize [um]')
plt.title('Displacement per 20 seconds')
plt.legend(title='Polarity window')
plt.show()

for strain in strains:
    for phase in hue_order:
        subset_data = final_df[(final_df['strain'] == strain) & (final_df['phase'] == phase)]['stepsize']
        median = np.median(subset_data)
        std_error_median = bootstrap_median(subset_data)
        print(f"Strain: {strain}, Phase: {phase}, Median: {median}, Standard Error of Median: {std_error_median}")




#################
# Define the color for the box plot
boxplot_color = 'dimgray'
hue_order = ['T1', 'T2']
strains=['', '', '']

# Convert the column to a categorical type with the custom order
final_df['column'] = pd.Categorical(final_df['strain'], categories=strains, ordered=True)
# Sort the DataFrame by the column
final_df = final_df.sort_values('strain')

# Define the order of strains
strains = ['', '', ' ']

# Merge num_files_per_strain with final_df
final_df = pd.merge(final_df, num_files_per_strain, on='strain', how='left')

# Drop the 'in_window' column if not needed anymore
final_df = final_df.drop(columns=['in_window'])

# Define hue order
hue_order = ['T1', 'T2']

# Define the color for the box plot
boxplot_color = 'dimgray'

# Create the violin plot
sns.violinplot(data=final_df, y='stepsize', x='strain', hue='phase', hue_order=hue_order, palette='Set3', order=strains)

# Add box plot with only the outline and specify the color
sns.boxplot(data=final_df, y='stepsize', x='strain', hue='phase', hue_order=hue_order, dodge='auto',
            linewidth=1.3, flierprops=dict(marker='x', markersize=5), fill=None, legend=False,
            gap=0.6, order=strains,
            palette={window: boxplot_color for window in hue_order})

# Annotate strain names with number of files
for i, row in num_files_per_strain.iterrows():
    plt.text(i, -0.05, f"(N={row['num_files']})", ha='center',size=8)

# Calculate and annotate p-values between conditions within each strain
for strain in strains:
    data_strain = final_df[final_df['strain'] == strain]
    windows = data_strain['phase'].unique()

    if len(windows) == 2:  # Ensure there are two groups to compare
        data_in_window = data_strain[data_strain['phase'] == hue_order[0]]['stepsize']
        data_out_window = data_strain[data_strain['phase'] == hue_order[1]]['stepsize']

        if len(data_in_window) > 1 and len(
                data_out_window) > 1:  # Check if there is enough data for the statistical test
            _, p_value = stats.ttest_ind(data_in_window, data_out_window)
            p_value_text = f'p={p_value:.3f}'

            # Calculate the position of p-value annotation based on the strain index
            strain_index = list(strains).index(strain)
            pos_x = strain_index
            pos_y = final_df[final_df['strain'] == strain][
                        'stepsize'].max() * 1.05  # Position above the highest data point

            plt.text(pos_x, pos_y, p_value_text, ha='center', va='center', size=10, alpha=0.75)

            # Add brackets (horizontal)
            style = '|-|'
            b_y = pos_y - (0.05 * pos_y)
            bracket = FancyArrowPatch(posA=(pos_x - 0.2, b_y), posB=(pos_x + 0.2, b_y), arrowstyle=style,
                                      mutation_scale=3, color='black', lw=1, linestyle='-', alpha=0.75)
            plt.gca().add_patch(bracket)

plt.ylabel('Stepsize [μm]')
plt.title('Displacement per 20 seconds')
plt.legend(title='Phase')
plt.tight_layout()
plt.show()


############### MSD
if MSD_inwindow is True:
    movement_df = movement_df[movement_df['in_window'] == True]

# Group by 'strain', 'time'
grouped = movement_df.groupby(['strain', 'time'])

# Calculate the mean and standard deviation for 'SD' and count of samples
mean_SD = grouped['SD'].agg(['mean', 'std', 'count']).reset_index()

# Rename columns
mean_SD.rename(columns={'mean': 'MSD', 'std': 'std', 'count': 'n_samples'}, inplace=True)

# filter data
    # only show n => 10
mean_SD = mean_SD[mean_SD['n_samples'] >= MSD_minN]

# mean_SD['MSD'] = np.log(mean_SD['MSD'])

def exponential_fit_function(x, A, B):
    """Exponential fitting function going through the origin."""
    return A * (np.exp(B * x) - 1)

def sigmoid(x,a,b,c):
    return a/(b + np.exp(-c*x))

############### MSD NEW
def plot_line_through_points(point1, point2, color='black', linestyle='-', label='Line through points'):
    """Plot a straight line through two points."""
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, color=color, linestyle=linestyle, label=label)


##################
# mean_SD = mean_SD[mean_SD['strain']=='WT']

plot_strains_in_separate_figures(data=mean_SD, x='time', y='MSD',
                                 # strains=['WT', 'rsr1Δ', 'axl2Δ rax1Δ'],
                                strains=['WT','WT','WT'],
                                 palette='Set2',
                                 fit_func='linear_window', fit_params=(0, 30, 20, 100), slope_unit='μm²/min',
                                 # length is in frames not minutes
                                 show_line=True, show_conf=True,
                                 title='', xlab='time [min]', ylab='MSD [μm²]',
                                 )
plt.show()

################### plot all paths
# Set the style and figure size
sns.set_style("whitegrid")
plt.figure(figsize=(8, 16))

# Get unique strains
unique_strains = movement_df['strain'].unique()

# Calculate the number of subplots needed
num_subplots = len(unique_strains)

# Calculate number of rows and columns for subplots
num_rows = (num_subplots + 1) // 2  # Add 1 to round up
num_cols = 3

# Create subplots
for i, strain in enumerate(unique_strains, start=1):
    plt.subplot(num_rows, num_cols, i)
    strain_data = movement_df[movement_df['strain'] == strain]
    sns.lineplot(data=strain_data, x='time', y='SD', hue='file', palette='icefire', alpha=0.6, legend=False)
    plt.xlabel('Time [min]')
    plt.ylabel('Squared displacement [μm²]')
    plt.title(f'Strain: {strain}')

# Adjust layout to avoid overlapping
plt.tight_layout()

plt.xlim([0,12])
plt.ylim([0,2])

# Move legend outside the plot and adjust layout
# plt.legend(title='File', bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.suptitle('Squared displacement over time')

# Show the plot
plt.show()

########## plot single to check files
# Set the style
sns.set_style("whitegrid")

# Get unique strains
unique_strains = movement_df['strain'].unique()

# Define the maximum number of lines per plot
max_lines_per_plot = 10

# Loop through unique strains
for strain in unique_strains:
    strain_data = movement_df[movement_df['strain'] == strain]
    num_lines = len(strain_data['file'].unique())

    # If the number of lines exceeds the maximum, split into multiple plots
    if num_lines > max_lines_per_plot:
        num_plots = (num_lines + max_lines_per_plot - 1) // max_lines_per_plot  # Calculate the number of plots needed
        files = strain_data['file'].unique()
        split_files = [files[i:i + max_lines_per_plot] for i in range(0, len(files), max_lines_per_plot)]

        # Create separate plots for each set of files
        for i, files_subset in enumerate(split_files):
            plt.figure(figsize=(8, 6))
            subset_data = strain_data[strain_data['file'].isin(files_subset)]
            sns.lineplot(data=subset_data, x='time', y='SD', hue='file', palette='Spectral')
            plt.xlabel('Time')
            plt.ylabel('SD')
            plt.title(f'SD over Time for Strain: {strain} (Part {i + 1})')
            plt.legend(title='File')
            plt.show()

    # If the number of lines is within the maximum, plot them all in one figure
    else:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=strain_data, x='time', y='SD', hue='file', palette='Spectral')
        plt.xlabel('Time')
        plt.ylabel('SD')
        plt.title(f'SD over Time for Strain: {strain}')
        plt.legend(title='File')
        plt.show()
