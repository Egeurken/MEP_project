"""
plots 3D plot of labelled mask with time slider, where different objects can be distinguished by colour.

input:
- run_direc: if True runs code for directory, if False only single file
- image_path: location of labelled mask file you want to visualise
- directory: directory for the labelled masks you want to visualise

output: 3D plot with slider, not being saved at the moment
"""

from skimage.io import imread
import os
from lib.label_and_analyse_plotting  import plot_with_slider_3D


run_direc = True  # if True runs code for plotting a directory, if False plots a single file

# plot a single file
if run_direc == False:
    image_path = ""
    image_data = imread(image_path)
    plot_with_slider_3D(image_data, image_path)

# plot a directory
else:
    directory = ""
    image_files = os.listdir(directory)

    # loop over files to plot all one by one
    for image in image_files:
        # opening files
        image_path = directory + image
        image_data = imread(image_path)

        name = image[10:-4]
        print(name)

        # plotting
        plot_with_slider_3D(image_data, name)

        # Wait for the plot to be closed before proceeding to the next image
        input("Press Enter to open next video...")  # comment this out if you want to open all at once
