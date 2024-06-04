from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import stats, cdist
from skimage import io, measure
from sklearn.decomposition import PCA



def mask_polarity(image, threshold_multiplier):
    """
    Make a masked version of a 3d cell image by applying an intensity threshold, which scales with the average intensity
    of the full image.
    :param image: timelapse of 3d stack containing a single cell
    :param threshold_multiplier: masking threshold is calculated as average * threshold_multiplier
    :return: thresholded image, where every pixel below the threshold is set to 0
    """
    if len(image.shape) > 3:
        image_start = image[0:10]
        average = np.average(image_start[image_start>0])
    else:
        average = np.average(image[image > 0])
    threshold = average * threshold_multiplier

    image[image < threshold] = 0

    return image


def label_objects(masked_image):
    """
    Converts a mask to a binary mask and then finds groups of nonzero values, assigning them distinct labels (1, 2, 3, 4, ...).

    Parameters:
    masked_image (list of ndarray): A list of 2D arrays representing frames, where each frame is a masked image.

    Returns:
    tuple: A tuple of 2D arrays, each containing labeled regions of the corresponding input frame.
    """
    mask_label = []

    # Loop over frames
    for j, frame in enumerate(masked_image):
        # Convert mask to binary mask
        frame[frame > 0] = 1

        # Generate labels for the frame
        frame_label, num_label = measure.label(frame, return_num=True, connectivity=3)

        # Add labeled frame to list
        mask_label.append(frame_label)

    # Convert output to tuple
    mask_label = tuple(mask_label)

    return mask_label


def get_com(mask_label, image_path):
    """
    Extracts a pixel within each labeled object in the mask.
    :param mask_label: labelled mask
    :param image_path: path of cropped_cell image
    :return: coordinates of pixels within each labeled object
    """
    #open image
    image = io.imread(image_path, plugin="tifffile")

    # store the pixel locations
    loc = []

    # loop over time
    for time, frame in enumerate(mask_label):
        # generate region properties
        region_props = measure.regionprops(frame, intensity_image=image[time], cache=False)
        com_frame = []

        # loop over objects
        for props in region_props:

            # get location of a pixel in the object
            middle = int(len(props.coords)/2)
            com_object = props.coords[middle]
            com_frame.append(com_object)
        loc.append(com_frame)
    return loc


def connect_labels_no_valley(mask_label, directory, max_valley):
    """
    Connects labels in the mask that are connected at the given threshold.

    Parameters:
    mask_label (tuple of ndarray): Labeled mask, where each element is a 2D array of labeled regions.
    directory (str): Directory containing the images.
    max_valley (int): Threshold for connecting labels.

    Returns:
    tuple of ndarray: Updated labeled mask with objects connected.
    """
    # Initialize new mask label
    mask_label_new = list(mask_label)

    # Get the center of mass for each labeled object
    com = get_com(mask_label, directory)

    # Read images from the directory
    image = io.imread(directory, plugin="tifffile")

    # Apply a polarity mask to the images
    masked_image_low = mask_polarity(image, max_valley)

    # Label objects in the low polarity mask
    mask_label_low = label_objects(masked_image_low)

    # Loop over each frame in the image sequence
    for time in range(len(image)):
        labels = []

        # Loop over each object in the frame
        for i, obj in enumerate(com[time]):
            obj = [int(x) for x in obj]
            i += 1

            if obj:
                # Get the label from the low polarity mask
                label_low = mask_label_low[time][obj[0]][obj[1]][obj[2]]

                if label_low in labels:
                    # Connect labels if already present
                    index = labels.index(label_low) + 1
                    mask_label_new[time][mask_label_new[time] == i] = index
                    labels.append(0)
                else:
                    # Append new label
                    labels.append(label_low)

    # Convert updated mask labels to tuple
    mask_label_new = tuple(mask_label_new)

    return mask_label_new


def connect_close_objects(mask, min_distv):
    """
    Combines spots for which the minimum distance between voxels is closer than min_distv.

    Parameters:
    mask (tuple of ndarray): Labeled mask, where each element is a 2D or 3D array of labeled regions.
    min_distv (float): Minimum distance threshold for merging objects.

    Returns:
    tuple of ndarray: Labeled mask with close objects merged.
    """
    masknp = np.array(mask)
    maskls = list(mask)

    # Loop over time
    for time in range(len(mask)):
        # Get the unique foreground values
        foreground_values = np.unique(masknp[time])

        # Remove background value (assuming it's 0)
        foreground_values = foreground_values[foreground_values != 0]

        # Get the coordinates of the foreground voxels for each label
        coords_dict = {value: np.array(np.where(masknp[time] == value)).T for value in foreground_values}

        # Iterate through pairs of labels
        for i, value1 in enumerate(foreground_values):
            for value2 in foreground_values[i + 1:]:
                # Find the closest voxels between the two components
                dist_matrix = cdist(coords_dict[value1], coords_dict[value2])
                min_dist = np.min(dist_matrix)

                # If the distance is less than min_distv, merge the components
                if 0 < min_dist < min_distv:
                    # Merge the labels by assigning all voxels of value2 to value1
                    maskls[time][maskls[time] == value2] = value1

    return tuple(maskls)


def remove_small_spots(mask_label, min_size):
    """
    Removes small spots from the labeled mask.

    Parameters:
    mask_label (tuple of ndarray): Labeled mask, where each element is a 2D or 3D array of labeled regions.
    min_size (int): Minimum size threshold for retaining spots (in voxels).

    Returns:
    tuple of ndarray: Labeled mask without small spots.
    """
    new_labelled_mask = []

    # Loop over time frames
    for time, frame in enumerate(mask_label):
        removed = []

        # Generate region properties
        region_props = measure.regionprops(frame, cache=False)

        # Loop over each region
        for props in region_props:
            size = props.area  # Using 'area' instead of 'num_pixels' for compatibility
            label = props.label

            # Remove regions smaller than the minimum size
            if size < min_size:
                frame[frame == label] = 0
                removed.append(label)

        # Reverse the removed labels list
        removed_r = removed[::-1]

        # Relabel the remaining regions
        for r in removed_r:
            if r < np.max(frame):
                for label in range(r + 1, np.max(frame) + 1):
                    frame[frame == label] = label - 1

        new_labelled_mask.append(frame)

    return tuple(new_labelled_mask)

def yaw(theta):
    """
    Returns the yaw rotation matrix for a given angle in degrees.

    Parameters:
    theta (float): The angle in degrees to rotate around the z-axis.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix representing yaw.
    """
    theta = np.deg2rad(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def pitch(theta):
    """
    Returns the pitch rotation matrix for a given angle in degrees.

    Parameters:
    theta (float): The angle in degrees to rotate around the y-axis.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix representing pitch.
    """
    theta = np.deg2rad(theta)
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def roll(theta):
    """
    Returns the roll rotation matrix for a given angle in degrees.

    Parameters:
    theta (float): The angle in degrees to rotate around the x-axis.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix representing roll.
    """
    theta = np.deg2rad(theta)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]
    ])


def bounding_box(mask):
    """
    Calculate the dimensions, angle, and center position of a bounding box that tightly encloses the True values in a binary mask.

    Parameters:
    mask (ndarray): A binary mask represented as a numpy array, where True values represent foreground elements.

    Returns:
    tuple:
        - length_sorted (list): A list containing the sorted lengths of the sides of the bounding box in descending order.
        - angle (float): The angle of the bounding box in degrees.
        - center (ndarray): The center position of the bounding box.

    The function first identifies the coordinates of all True values in the mask. It then performs Principal Component Analysis (PCA)
    to find the principal axes of the True values. Using the principal axes, the function calculates the center of the bounding box
    and projects all True points onto the principal axes to align them with the bounding box axes. Finally, it computes the dimensions
    of the bounding box based on the maximum and minimum coordinates along each axis and returns the sorted lengths of the sides,
    the angle of the bounding box, and its center position.

    Note: This function assumes the mask represents a binary segmentation mask of a 3D volume.
    """
    # Find the coordinates of all True values in the mask
    true_indices = np.argwhere(mask)

    # Perform PCA to find principal axes
    pca = PCA(n_components=3)
    pca.fit(true_indices)
    principal_axes = pca.components_

    # Calculate the angle of the bounding box
    angle = np.arctan2(principal_axes[0][1], principal_axes[0][0]) * (180 / np.pi)

    # Find the center of the bounding box
    center = np.mean(true_indices, axis=0)

    # Project all points onto the principal axes
    projected_points = np.dot(true_indices - center, principal_axes.T)

    # Get the minimum and maximum coordinates along each axis
    min_coords = np.min(projected_points, axis=0)
    max_coords = np.max(projected_points, axis=0)

    # Calculate side lengths
    side_lengths = max_coords - min_coords

    # Sort side lengths in descending order
    length_sorted = sorted(side_lengths, reverse=True)

    return length_sorted, angle, center

def polygon_surface(masked_image):
    """
    Calculate the surface area of the polygon(s) enveloping all non-zero pixels over time.

    Parameters:
    masked_image (ndarray): Timelapse of thresholded images containing non-zero pixels of interest.

    Returns:
    ndarray: Surface area over time.
    """
    # Define cross product function to avoid potential NumPy bug
    cross = lambda x, y: np.cross(x, y)

    # Initialize arrays to hold the volume and surface area
    poly_surface = np.zeros(len(masked_image))

    # Check if the masked image contains any non-zero pixels
    if not np.all(masked_image == 0):
        # Apply the marching cubes algorithm to extract the mesh
        vertices, faces, normals, values = measure.marching_cubes(masked_image, 0)
        triangles = vertices[faces]

        # Calculate the surface area of the mesh
        for triangle in triangles:
            p1, p2, p3 = triangle[0], triangle[1], triangle[2]
            normal = cross(p3 - p1, p2 - p1)
            surface = 0.5 * np.linalg.norm(normal)
            poly_surface += surface

    return poly_surface


def start_intensity(image):
    """
    Calculate the average intensity of the initial frames of the image.

    Parameters:
    image (ndarray): Input image, which can be a 3D or 4D array.

    Returns:
    float: Average intensity of the initial frames.
    """
    # Check if the image has more than 3 dimensions (4D)
    if len(image.shape) > 3:
        # Extract the first 10 frames for a 4D image
        image_start = image[0:10]
        # Calculate the average intensity of non-zero values
        average = np.average(image_start[image_start > 0])
    else:
        # Calculate the average intensity of non-zero values for a 3D image
        average = np.average(image[image > 0])

    return average


def remove_missing_labels(mask):
    """
    Remove missing labels from the mask.

    Parameters:
    mask (list of ndarray): List of 2D or 3D numpy arrays representing labeled masks.

    Returns:
    list of ndarray: Mask without missing labels.
    """
    mask_new = []

    # Loop over time frames
    for time, frame in enumerate(mask):
        max_label = np.max(frame)
        for i in range(max_label, 0, -1):
            if i not in frame:  # Check if label i is present
                # Lower labels larger than i by one
                frame[frame > i] -= 1

        mask_new.append(frame)

    return mask_new

def total_intensity(mask, image):
    """
    Calculate the total intensity of the image where the mask is 1.

    Parameters:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask with values 0 and 1.

    Returns:
        float: Total intensity of the image where the mask is 1.
    """
    # Ensure image and mask have the same dimensions
    assert image.shape == mask.shape, "Image and mask dimensions do not match"

    # Multiply the image with the mask and sum the intensities where mask is 1
    total_intensity = np.sum(image[mask == 1])

    return total_intensity


def get_properties(mask, image_path, start, end, direc, start_int, voxel_size):
    """
    Compute properties of objects and save the results.

    Parameters:
        mask (list of ndarray): Input mask.
        image_path (str): Path of the image.
        start (int): Start time.
        end (int): End time.
        direc (str): Directory.
        start_int (float): Starting intensity.
        voxel_size (float): Voxel size.

    Returns:
        pd.DataFrame: DataFrame with properties.
    """
    image = io.imread(image_path, plugin="tifffile")

    prop_table = pd.DataFrame(columns=['strain', 'time', 'in_window', 'n_spots', 'volume', 'surface', 'centroid',
                                       'intensity_max', 'total_int', 'bbox', 'bbox_angle', 'bbox_center'])

    for time in range(len(mask)):
        frame = mask[time]  # Access the correct frame from mask_fil_label

        # If the frame is empty then there are no properties to compute
        if np.max(frame) == 0:
            continue

        region_props = measure.regionprops(frame, intensity_image=image[time], cache=False,
                                           extra_properties=(total_intensity, polygon_surface, bounding_box))

        n_spots = len(region_props)
        in_window = start <= time < end

        for prop in region_props:
            n_pix = prop.area * voxel_size
            int_max = prop.max_intensity / start_int
            cen = prop.weighted_centroid

            total_int = prop.total_intensity
            surface = prop.polygon_surface * voxel_size ** (2 / 3)
            bbox, bbox_angle, bbox_center = prop.bounding_box

            # These are other properties that are possible to include
            # euler = prop.euler_number
            # inertia_tens = prop.inertia_tensor
            # inertia_tens_eig = prop.inertia_tensor_eigvals
            # extent = prop.extent
            # solid = prop.solidity
            # int_mean = prop.mean_intensity / start_int
            # major = prop.major_axis_length
            # minor = prop.minor_axis_length

            new_row = {'strain': direc, 'time': time, 'in_window': in_window, 'n_spots': n_spots, 'volume': n_pix,
                       'surface': surface, 'centroid': cen, 'intensity_max': int_max, 'total_int': total_int,
                       'bbox': bbox, 'bbox_angle': bbox_angle, 'bbox_center': bbox_center}
            prop_table = prop_table.append(new_row, ignore_index=True)

    return prop_table


