"""
This script defines functions for simulating and analyzing 2D Brownian motion and plotting mean squared displacement (MSD).

Input:
    num_steps: Number of steps for the simulation.
    mean_step: Mean step length.
    iterations: Number of iterations for the simulation.
    ring_shape: Boolean indicating whether to confine the motion to a ring shape.
    inner_radius: Inner radius of the ring shape.
    outer_radius: Outer radius of the ring shape.
    strain: Strain name for the simulation.

Output:
    squared_displacement: DataFrame containing the squared displacement data.
    mean_squared_displacement: DataFrame containing the mean squared displacement data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def simulate_and_analyze_2d_brownian_motion(num_steps, mean_step, iterations, ring_shape=False, inner_radius=0, outer_radius=1, strain='rsr1d'):
    """
    Simulates and analyzes 2D Brownian motion.

    Parameters:
        num_steps (int): Number of steps for the simulation.
        mean_step (float): Mean step length.
        iterations (int): Number of iterations for the simulation.
        ring_shape (bool): Indicates whether to confine the motion to a ring shape.
        inner_radius (float): Inner radius of the ring shape.
        outer_radius (float): Outer radius of the ring shape.
        strain (str): Strain name for the simulation.

    Returns:
        squared_displacement (DataFrame): DataFrame containing the squared displacement data.
        mean_squared_displacement (DataFrame): DataFrame containing the mean squared displacement data.
    """
    def distance_2d(point1, point2):
        return abs(np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

    def SD(path):
        SD = []
        start = path[0]

        for index, row in enumerate(path):
            current_position = row
            distance = distance_2d(start, current_position)
            SD.append(abs(distance**2))

        return SD

    def simulate_2d_brownian_motion(num_steps, mean_step, ring_shape=False, inner_radius=0, outer_radius=1):
        path = np.zeros((num_steps + 1, 2))
        current_position = [inner_radius, 0]

        for step in range(0, num_steps + 1):
            random_angle = np.random.uniform(0, 2 * np.pi)
            random_step_length = np.random.normal(mean_step, 0.05)
            random_step = [random_step_length * np.cos(random_angle), random_step_length * np.sin(random_angle)]

            new_position = [current_position[0] + random_step[0], current_position[1] + random_step[1]]

            if ring_shape:
                distance_from_origin = np.linalg.norm(new_position)
                while distance_from_origin < inner_radius or distance_from_origin > outer_radius:
                    random_angle = np.random.uniform(0, 2 * np.pi)
                    random_step_length = np.random.normal(mean_step, 0.05)
                    random_step = [random_step_length * np.cos(random_angle), random_step_length * np.sin(random_angle)]
                    new_position = [current_position[0] + random_step[0], current_position[1] + random_step[1]]
                    distance_from_origin = np.linalg.norm(new_position)

            current_position = new_position
            path[step] = current_position

        return path

    SDs = pd.DataFrame(columns=['time', 'SD', 'iteration'])

    for ii in range(iterations):
        path = simulate_2d_brownian_motion(num_steps, mean_step, ring_shape=ring_shape, inner_radius=inner_radius, outer_radius=outer_radius)
        sd_values = SD(path)

        for idx, sd in enumerate(sd_values):
            time = idx / 3
            SDs = SDs.append({'SD': sd, 'time': time, 'iteration': ii, 'strain': strain}, ignore_index=True)

    grouped = SDs.groupby(['time', 'strain'])
    mean_SD = grouped['SD'].agg(['mean', 'std', 'count']).reset_index()
    mean_SD.rename(columns={'mean': 'MSD', 'std': 'Standard_Deviation', 'count': 'Number_of_samples'}, inplace=True)

    return SDs, mean_SD


# Example usage:
num_steps = 0
mean_step = 0
iterations = 0
inner_radius = 0
outer_radius = 0
ring_shape = True
strain = ''

squared_displacement, mean_squared_displacement = simulate_and_analyze_2d_brownian_motion(num_steps, mean_step, iterations, ring_shape=ring_shape, inner_radius=inner_radius, outer_radius=outer_radius, strain=strain)


def plot_mean_squared_displacement(mean_squared_displacement):
    """
    Plots the mean squared displacement over time.

    Parameters:
        mean_squared_displacement (DataFrame): DataFrame containing mean squared displacement data.
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_squared_displacement['time'], mean_squared_displacement['MSD'], yerr=mean_squared_displacement['Standard_Deviation'], fmt='-o', capsize=5)
    plt.xlabel('Time [min]')
    plt.ylabel('Mean Squared Displacement [um²]')
    plt.title('Mean Squared Displacement over Time')
    plt.grid(True)
    plt.show()


# Example usage:
plot_mean_squared_displacement(mean_squared_displacement)
