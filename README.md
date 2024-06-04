# Labelling Module

The labelling module is designed for processing images, mainly focusing on generating labeled masks and analyzing 
properties of labeled objects. It takes `cropped_cells` as input to create a mas by thresholding. This mask is then labeled, 
assigning connected voxels with the same value, which effectively segments objects in the image. The module allows for 
filtering or merging of objects within the mask. Finally, it computes properties of all spots for each frame, enabling 
detailed analysis.

## Features

### Labelling Main

- **Mask Generation:** Generates a mask using thresholding.
- **Labeling:** Assigns connected pixels within the mask the same value, enabling segmentation of objects.
- **Connecting Close Objects:** Two objects closer than a minimum distance are assigned the same label.
- **Connecting Objects:** Connect objects which are connected with an intensity above a certain threshold.
- **Filtering Objects:** Can filter out objects based on properties such as size.
- **Property Computation:** Computes properties for objects over time.
- **Counting Number of Spots**

### Other

- **Visualizing Labelled Masks**

### Analyse Properties

- **Plot Properties Over Time**
- **Compute Properties of Spot in Final Frame of Time Window**
- **Compute mean properties over time**
- **Plot duration of polarity establishment**

### Spot Movement

- **Track the Movement of the Spot in Polarity Window to Compute Step Size and MSD**
- **Modeling a 2D Random Walk in Absence and Presence of Spatial Cues**
- **Compare model and experimental data**

## Usage

1. Generate `cropped_cells` (outside module).
2. Run `labelling_main` (saves properties and labelled masks).
3. Analyse labelled masks/properties using one of the scripts.

note: the files label_and_analyse and label_and_analyse_plotting should be present in lib folder in order to 
execute the code

## Environment Variables

- **Environment:** Post_Segmentation
- **Python:** 3.11

**Dependencies:**
- `view_labelled_masks` requires Plotly (version: 5.19.0).
- requires Seaborn (version: 0.13.2).

## Useful Documentation

- **Labelling:** [scikit-image label documentation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label)
- **Computing Properties:** [scikit-image regionprops documentation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
