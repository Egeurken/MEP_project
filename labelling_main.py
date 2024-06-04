"""
plots 3D plot of labelled mask with time slider, where different objects can be distinguished by colour.

input:
- run_direc: if True runs code for directory, if False only single file
- image_path: location of labelled mask file you want to visualise
- directory: directory for the labelled masks you want to visualise

output: 3D plot with slider, not being saved at the moment
"""

import os
import numpy as np
from skimage import io

from lib.timeseries_processing import list_to_array
from lib.label_and_analyse import (get_properties, label_objects, mask_polarity, connect_labels_no_valley,
                                   remove_small_spots, start_intensity, remove_missing_labels, connect_close_objects)
from lib.process_3d_cell import get_polar_endpoint


def create_folders_if_not_exists(folders_and_locations, base_location=""):
    """
    Create folders if they do not exist.

    Args:
        folders_and_locations (dict): A dictionary mapping folder names to their locations.
        base_location (str, optional): The base location to prepend to the folder locations. Defaults to "".
    """
    for folder, location in folders_and_locations.items():
        if base_location:
            folder_path = os.path.join(base_location, location, folder)
        else:
            folder_path = os.path.join(location, folder)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder}' created in '{folder_path}'.")
        else:
            print(f"Folder '{folder}' in '{folder_path}' already exists.")
    print()

#################################### In and Output locations ###########################################
base = ''  # Base location
directory_set = []  # Directories containing image files
category_labels = []  # Labels for categories
Experiment = []  # Experiment numbers
output_directory = ''  # Output directory containing subfolders for labelled masks and properties

######################## Parameters ########################
# Imaging properties
scale = 0  # um/pixel
voxel_size = scale ** 3  # um3

# Threshold
threshold = 0  # Based on average start intensity

# Filtering parameters
min_size = 0  # Minimum size of spot to be included (in um3)
max_valley = 0  # All objects that are connected when thresholded at this valley will be connected
min_dist = 0  # Minimum smallest distance for which objects won't be combined (in um)

######################## CODE ########################
total = []
dur = []
n = [0] * len(directory_set)
sizes = [[], []]
sizes_big = [[], []]
max_naive = [[], []]
cooc_time = [0, 0, 0]

current_category_label = None
current_experiment_number = None
current_directory = None

# Check if folders for saving exist
info = f'_T={threshold},Tmin={max_valley}'

# Define folders and subfolders
folders = ['labelled_masks' + info, 'properties' + info]
subfolders = ['1_spot', '2_spots']

# Define folders and their respective locations
folders_and_locations = {
    folders[0]: output_directory,
    folders[1]: output_directory,
    os.path.join(folders[0], '1_spot'): output_directory,
    os.path.join(folders[0], '2_spots'): output_directory,
    os.path.join(folders[1], '1_spot'): output_directory,
    os.path.join(folders[1], '2_spots'): output_directory,
}

# Call the function to create folders if they do not exist
create_folders_if_not_exists(folders_and_locations)

for d, directories in enumerate(directory_set):
    for j, directory in enumerate(directories):
        # Update experiment label
        E = Experiment[d][j]

        os.system('cls')
        print(f"\r Working on directory: {category_labels[int(d)]} : {E}")

        # Open directories
        image_files = os.listdir(directory)
        n_files = len(image_files)

        # Count the number of files
        n[d] += n_files
        two_spots = [0] * n_files

        for i, image_file in enumerate(image_files):

            print(f"\r Processing: {image_file}")

            # Open cell videos
            image_path = directory + image_file
            image = io.imread(image_path, plugin="tifffile")

            # Get image start intensity (for later normalization)
            start_int = start_intensity(image)

            # Generate mask
            masked_image = mask_polarity(image, threshold)

            # Determine polarity window (for True/False in dataframe)
            polar_start, polar_end = get_polar_endpoint(masked_image)

            # Label mask
            mask_label = label_objects(masked_image)

            # Remove objects closer than min_dist
            try:
                mask_label1 = connect_close_objects(mask_label, min_dist / scale)
            except np.core._exceptions._ArrayMemoryError:
                print('Memory error: Unable to allocate memory for distance matrix.')
                continue

            # Connect labelled objects where there is no valley
            mask_label2 = connect_labels_no_valley(mask_label1, image_path, max_valley)

            # Remove objects smaller than min_size
            mask_label3 = remove_small_spots(mask_label2, min_size / voxel_size)

            # Make sure that n labels represents n objects
            mask_label4 = remove_missing_labels(mask_label3)

            try:
                properties = get_properties(mask_label4, image_path, polar_start, polar_end, category_labels[int(d)],
                                            start_int, voxel_size)
            except ValueError as e:
                print(f"Skipping index {i} due to ValueError: {e}")
                # I think this means that the frame is empty
                continue

            # Add experiment number to properties
            properties['E'] = E

            # Count the number of spots
            two_spots = False
            if np.max(mask_label) >= 2:
                two_spots = True

            # Saving 1 and 2 spots to separate folders
            if two_spots:
                print('2_spots')
                # Labelled masks
                mask_fil_label = list_to_array(mask_label)
                output_direct = os.path.join(output_directory, 'labelled_masks' + info + '/2_spots')
                filename = os.path.join(output_direct,
                                        'labelled_' + str(image_file[7:-4]) + '_' + category_labels[int(d)] + '.tif')
                io.imsave(filename, mask_fil_label, plugin="tifffile", imagej=True,
                          resolution=(1 / scale, 1 / scale),
                          metadata={'spacing': 1 / scale, 'unit': 'um', 'axes': 'TZYX', 'fps': 0.05},
                          check_contrast=False)
                # Properties
                output_direct = os.path.join(output_directory, 'properties' + info + '/2_spots')
                filename = os.path.join(output_direct,
                                        'props' + str(image_file[7:-4]) + '_' + category_labels[int(d)] + '.csv')
