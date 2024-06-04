import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use the darkgrid theme for seaborn
sns.set_theme(style="darkgrid")


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


# Directory containing the data files
directory = ''
direc = os.listdir(directory)

# Dataframe to hold the properties
properties = pd.DataFrame(columns=['strain', 'time', 'window'])

for file in direc:
    df = pd.read_csv(os.path.join(directory, file))

    if len(df) >= 5:
        # Keep only part dataframe where there is co-occurrence
        start, end = cooc_start_end(df)
        filtered_df = df[(df['time'] >= start) & (df['time'] <= end)]

        # Add properties to new dataframe
        if start and end:
            time = 1 + end - start  # Duration of 2 spots
            strain = df.loc[1, 'strain']
            window = in_window(filtered_df)

            new_row = {'strain': strain, 'time': time, 'window': window}
            properties = properties._append(new_row, ignore_index=True)

print(properties)

# Plot window and timing of spots
# Define a color palette based on the unique strains in your dataset
palette = sns.color_palette("husl", len(properties['strain'].unique()))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# First Plot: In or after polarity window
ax[0].set_title('In or after polarity window')
sns.countplot(data=properties, x='strain', hue='window', ax=ax[0], palette=palette)

# Second Plot: Co-occurrence time [frames]
ax[1].set_title('Co-occurrence time [frames]')
sns.violinplot(data=properties, y='time', x='strain', ax=ax[1], cut=0, palette=palette, alpha=0.5)
sns.swarmplot(data=properties, y='time', x='strain', ax=ax[1], palette=palette)

plt.show()
