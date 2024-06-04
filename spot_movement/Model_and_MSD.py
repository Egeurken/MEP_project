"""
This script processes data from property files to compute step sizes and mean squared displacement (MSD).
It also includes functions for plotting step size and MSD data, as well as simulating and analyzing 2D Brownian motion
to generate model data for comparison.

Input:
    directory: Directory containing property files.
    max_stepsize: Determines the maximum step size to be included in the analysis.
                  Steps larger than this value are considered outliers.
    MSD_inwindow: Determines whether to make the MSD plot for the full window or only part.
                  If False, the step size plot is affected.
    MSD_minN: Denotes how many samples must be present for a timepoint to be included in the MSD plot.
    plot: Indexes of paths for which a 3D plot will be made.
    scale: Scale of imaging (coordinates are still in voxels, will change in next version).

Output:
    Step size plot, separated based on whether the data is in window.
    MSD plot, showing MSD, standard deviation, and the slope of the curve.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import vg

from lib.label_and_analyse_plotting import plot_strains_in_separate_figures
from Model_MSD import simulate_and_analyze_2d_brownian_motion


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
    p1 = np.array(p1)  # Convert tuples to NumPy arrays for easy calculation
    p2 = np.array(p2)

    vec1 = p2 - p1
    vec2 = np.array([1, 0, 0])

    angle = vg.angle(vec1, vec2, look=vg.basis.z, units='deg')
    if not acute:
        angle = angle if angle <= np.pi else 2 * np.pi - angle  # Adjust angle to be between 0 and 2*pi
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
    prev_position = parse_centroid(df.iloc[0]['centroid'])  # Initial position

    for i in range(1, len(df)):
        current_position = parse_centroid(df.iloc[i]['centroid'])
        distance = scale * distance_3d(prev_position, current_position)
        distances_prev.append(distance)
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

    x_actual = [parse_centroid(coord)[0] for coord in df['centroid']]
    y_actual = [parse_centroid(coord)[1] for coord in df['centroid']]
    z_actual = [parse_centroid(coord)[2] for coord in df['centroid']]

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


directory = ''
direc = os.listdir(directory)

max_stepsize = 0
MSD_inwindow = False
MSD_minN = 0
plot = []
scale = 0

movement_df = pd.DataFrame()
sns.set_theme(style="darkgrid")

for i, file in enumerate(direc):
    df = pd.read_csv(directory + file)

    if MSD_inwindow is False:
        df = df[df['in_window'] == False].reset_index(drop=True)

    if len(df) >= 5 and len(df) < 100:
        distances_prev = distance_previous_frame(df, scale)
        df['stepsize'] = distances_prev

        for index, row in df.iterrows():
            if row['stepsize'] > max_stepsize:
                if row['in_window']:
                    df = df.iloc[:index, :]
                else:
                    df = df.iloc[index + 1:, :]

        if len(df) < 5:
            continue

        time = list(range(len(df)))
        df['time'] = [(1/3) * t for t in time]
        df['file'] = file

        if i in plot:
            try:
                plot_path_with_actual(df, file)
            except KeyError as err:
                print(f"KeyError occurred: {err}")

        displacement = []
        start_pos = parse_centroid(df.iloc[0]['centroid'])
        for _, row in df.iterrows():
            current_position = parse_centroid(row['centroid'])
            distance = scale * distance_3d(start_pos, current_position)
            displacement.append(abs(distance))

        SD = [x**2 for x in displacement]
        df['SD'] = SD
        df_fil = df.filter(['strain', 'time', 'in_window', 'stepsize', 'SD', 'file'])
        combine = [movement_df, df_fil]
        movement_df = pd.concat(combine)

filtered_df = movement_df[movement_df['time'] != 0]
filtered_df = filtered_df[filtered_df['stepsize'] < 2]

if MSD_inwindow is True:
    movement_df = movement_df[movement_df['in_window'] == True]

grouped = movement_df.groupby(['strain', 'time'])
mean_SD = grouped['SD'].agg(['mean', 'std', 'count']).reset_index()
mean_SD.rename(columns={'mean': 'MSD', 'std': 'std',                              'count': 'n_samples'}, inplace=True)
mean_SD = mean_SD[mean_SD['n_samples'] >= MSD_minN]

strain_parameters = {
    '': {'num_steps': 0, 'mean_step': 0, 'iterations': 1, 'ring_shape': True, 'inner_radius': 0, 'outer_radius': 1},
    '': {'num_steps': 0, 'mean_step': 0, 'iterations': 1, 'ring_shape': True, 'inner_radius': 0, 'outer_radius': 1},
    '': {'num_steps': 0, 'mean_step': 0, 'iterations': 1, 'ring_shape': True, 'inner_radius': 0, 'outer_radius': 1}
}

results = {}
for strain, params in strain_parameters.items():
    _, mean_squared_displacement = simulate_and_analyze_2d_brownian_motion(**params, strain=strain)
    results[strain] = mean_squared_displacement

model_data = pd.concat([pd.DataFrame(results[strain]) for strain in results.keys()], ignore_index=True)
model_data['strain'] = [strain for strain in results.keys() for _ in range(len(results[strain]))]
model_data = model_data[['strain', 'time', 'MSD', 'Standard_Deviation', 'Number_of_samples']]

plot_strains_in_separate_figures(data=mean_SD, x='time', y='MSD', strains=list(results.keys()),
                                 palette='Set2',
                                 model=model_data, fit_func=None,
                                 show_line=True, show_conf=False,
                                 title='',
                                 xlab='Time [min]', ylab='MSD [um²]')

plt.show()

