import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Use the darkgrid theme for seaborn
sns.set_theme(style="darkgrid")

"""
This script processes tracking data for spots in cells, determines co-occurrence windows,
and visualizes the results. It involves reading CSV files, parsing centroids, tracking spot identities
across frames, and plotting the volumes of spots over time.
"""


def parse_centroid(centroid_str):
    """
    Parse a centroid string into a tuple of float coordinates.

    Args:
        centroid_str (str): The centroid string in the format '(x, y)'.

    Returns:
        tuple: A tuple containing the x and y coordinates as floats.
    """
    centroid_str = centroid_str.strip('()')  # Remove parentheses
    coordinates = centroid_str.split(',')  # Split by comma
    return tuple(float(coord) for coord in coordinates)


def cooc_start_end(df):
    """
    Determine the start and end times of co-occurrence of spots.

    Args:
        df (pd.DataFrame): The dataframe containing 'n_spots' and 'time' columns.

    Returns:
        tuple: A tuple containing the start and end times.
    """
    start_index = None
    end_index = None

    for index, row in df.iterrows():
        if row['n_spots'] == 2 and start_index is None:
            start_index = row['time']
        elif start_index is not None and row['n_spots'] == 2:
            end_index = row['time']

    return start_index, end_index


def in_window(df):
    """
    Determine the occurrence window status based on the first and last rows.

    Args:
        df (pd.DataFrame): The dataframe containing 'in_window' column.

    Returns:
        str: The window status: 'in', 'out', or 'partially'.
    """
    first_row = bool(df.iloc[0]['in_window'])
    last_row = bool(df.iloc[-1]['in_window'])

    if last_row is True and first_row is True:
        return "in"
    elif last_row is False and first_row is False:
        return 'out'
    else:
        return 'partially'  # Note this means the co-occurrence is at the edge


def initialize_spot_counter():
    """
    Initialize an empty DataFrame to keep track of spot labels and their centroids.

    Returns:
        pd.DataFrame: An empty DataFrame with columns 'label', 'last_seen', and 'centroid'.
    """
    return pd.DataFrame(columns=['label', 'last_seen', 'centroid'])


def assign_labels_first_time(time_frame, spot_counter, df, nlabel, index_used):
    """
    Assign labels to spots for the first time in a given time frame.

    Args:
        time_frame (pd.DataFrame): The dataframe for the current time frame.
        spot_counter (pd.DataFrame): The spot counter DataFrame.
        df (pd.DataFrame): The main dataframe with all spots.
        nlabel (int): The current label counter.
        index_used (list): A list of used labels.

    Returns:
        tuple: Updated spot_counter, df, nlabel, and index_used.
    """
    for index, row in time_frame.iterrows():
        nlabel += 1
        centroid = np.array([float(x) for x in row['centroid'][1:-1].split(',')])
        spot_counter = spot_counter._append({'label': nlabel, 'last_seen': row['time'], 'centroid': centroid},
                                            ignore_index=True)
        df.at[index, 'spot_identity'] = nlabel
        index_used.append(nlabel)

    return spot_counter, df, nlabel, index_used


def find_closest_indices(time_frame, spot_counter, max_distance):
    """
    Find the closest spot indices in the current time frame to those in the spot counter.

    Args:
        time_frame (pd.DataFrame): The dataframe for the current time frame.
        spot_counter (pd.DataFrame): The spot counter DataFrame.
        max_distance (float): The maximum allowable distance for matching spots.

    Returns:
        tuple: Closest indices, distances, and centroids arrays.
    """
    centroids = np.array(
        [np.array([float(x) for x in row['centroid'][1:-1].split(',')]) for _, row in time_frame.iterrows()])
    prev_centroids = np.array([row['centroid'] for _, row in spot_counter.iterrows()])
    distances = np.linalg.norm(centroids[:, None] - prev_centroids, axis=2)
    closest_indices = np.argmin(distances, axis=1)

    closest_distances = np.min(distances, axis=1)

    unique_indices, counts = np.unique(closest_indices, return_counts=True)
    duplicate_indices = unique_indices[counts > 1]

    for index in duplicate_indices:
        duplicate_mask = closest_indices == index
        if np.sum(duplicate_mask) > 1:
            duplicate_distances = closest_distances[duplicate_mask]
            min_distance_index = np.argmin(duplicate_distances)

            min_distance_indices = np.where(duplicate_distances == duplicate_distances[min_distance_index])[0]

            if len(min_distance_indices) > 1:
                min_distance_index = min(min_distance_indices)

            for i, val in enumerate(duplicate_mask):
                if val and i != min_distance_index:
                    closest_indices[i] = -1

    return closest_indices, distances, centroids


def assign_labels(time_frame, spot_counter, df, closest_indices, distances, centroids, max_distance, nlabel,
                  index_used, time):
    """
    Assign labels to spots based on the closest matching centroids.

    Args:
        time_frame (pd.DataFrame): The dataframe for the current time frame.
        spot_counter (pd.DataFrame): The spot counter DataFrame.
        df (pd.DataFrame): The main dataframe with all spots.
        closest_indices (np.array): The indices of the closest matching spots.
        distances (np.array): The distances between centroids.
        centroids (np.array): The centroids of the spots in the current time frame.
        max_distance (float): The maximum allowable distance for matching spots.
        nlabel (int): The current label counter.
        index_used (list): A list of used labels.
        time (int): The current time value.

    Returns:
        tuple: Updated spot_counter, df, nlabel, and index_used.
    """
    for i, closest_index in enumerate(closest_indices):
        if closest_index is not None and distances[i, closest_index] <= max_distance and closest_index != -1:
            df.at[time_frame.index[i], 'spot_identity'] = spot_counter.iloc[closest_index]['label']
            index_used.append(spot_counter.iloc[closest_index]['label'])
            spot_counter.loc[spot_counter['label'] == spot_counter.iloc[closest_index]['label'], 'last_seen'] = time
        else:
            nlabel += 1
            while nlabel in index_used:
                nlabel += 1
            index_used.append(nlabel)
            spot_counter = spot_counter._append(
                {'label': nlabel, 'last_seen': time_frame['time'].iloc[i], 'centroid': centroids[i]},
                ignore_index=True)
            df.at[time_frame.index[i], 'spot_identity'] = nlabel

    return spot_counter, df, nlabel, index_used


def tracking(df, max_frames_to_keep=10, max_distance=10):
    """
    Track spots across frames in a dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing spot data.
        max_frames_to_keep (int): The maximum number of frames to keep for tracking.
        max_distance (float): The maximum allowable distance for matching spots.

    Returns:
        tuple: The spot counter DataFrame and the updated input dataframe.
    """
    df['spot_identity'] = np.nan
    spot_counter = initialize_spot_counter()
    nlabel = 0
    index_used = []

    for time_value in df['time'].unique():
        time_frame = df[df['time'] == time_value]

        threshold = time_value - max_frames_to_keep
        spot_counter = spot_counter[spot_counter['last_seen'] >= threshold]

        if len(spot_counter) == 0:
            spot_counter, df, nlabel, index_used = assign_labels_first_time(time_frame, spot_counter, df, nlabel,
                                                                            index_used)
        else:
            closest_indices, distances, centroids = find_closest_indices(time_frame, spot_counter, max_distance)
            spot_counter, df, nlabel, index_used = assign_labels(time_frame, spot_counter, df, closest_indices,
                                                                 distances, centroids, max_distance, nlabel,
                                                                 index_used, time_value)

    return spot_counter, df


# Directory containing the data files
directory = ''

# List of files in the directory
direc = os.listdir(directory)

# Iterate over files in the directory
for file in direc:
    # Read the CSV file into a DataFrame
    print(file)
    df = pd.read_csv(directory + file)

    # Track spots in the DataFrame
    spot_counter, df = tracking(df, 5, 20)

    selected_columns = df.loc[:, ['time', 'n_spots', 'spot_identity']]
    two_spots = df[df['n_spots'] == 2]

    df['time'] = df['time'] - df['time'].iloc[0]
    df['time'] = df['time'] * (1 / 3)

    sns.lineplot(data=df, x='time', y='volume', hue='spot_identity', marker='o', palette='Set1')
    plt.title('')
    plt.xlabel('Time [min]')
    plt.ylabel('Volume per spot [µm³]')
    plt.show()
