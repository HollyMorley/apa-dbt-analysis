"""Side-by-side comparison of two DLC model outputs on the same video."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from helpers.config import *

# Load video and deeplabcut coordinates
video_path = paths["local_video_folder"] + r"\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_side_1.avi"
coord_path = {
    'old': paths["data_folder"] + r"\Round1\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_side_1DLC_resnet50_DLC_DualBeltJul25shuffle1_1030000.h5",
    'new': paths["data_folder"] + r"\Round2\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_side_1DLC_resnet50_DLC_DualBeltAug2shuffle1_600000.h5"
}

# Load data
data = {
    'new': pd.read_hdf(coord_path['new']),
    'old': pd.read_hdf(coord_path['old'])
}

# Drop the first level of the multi-index columns
data['new'] = data['new'].droplevel(0, axis=1)
data['old'] = data['old'].droplevel(0, axis=1)

# Define the range for clipping
lower = 209663
upper = 209909

# Clip data to the specified range
data['new'] = data['new'].loc[lower:upper]
data['old'] = data['old'].loc[lower:upper]

# Likelihood masks for each point
pcutoff = 0.9  # Assuming a cutoff value is given; replace with actual if different
fpToeR_mask = {version: data[version].loc[:, ('ForepawToeR', 'likelihood')] > pcutoff for version in ['new', 'old']}
fpKnuckleR_mask = {version: data[version].loc[:, ('ForepawKnuckleR', 'likelihood')] > pcutoff for version in ['new']}
fpAnkleR_mask = {version: data[version].loc[:, ('ForepawAnkleR', 'likelihood')] > pcutoff for version in ['new', 'old']}
fpToeL_mask = {version: data[version].loc[:, ('ForepawToeL', 'likelihood')] > pcutoff for version in ['new', 'old']}
fpKnuckleL_mask = {version: data[version].loc[:, ('ForepawKnuckleL', 'likelihood')] > pcutoff for version in ['new']}
fpAnkleL_mask = {version: data[version].loc[:, ('ForepawAnkleL', 'likelihood')] > pcutoff for version in ['new', 'old']}
hpToeR_mask = {version: data[version].loc[:, ('HindpawToeR', 'likelihood')] > pcutoff for version in ['new', 'old']}
hpKnuckleR_mask = {version: data[version].loc[:, ('HindpawKnuckleR', 'likelihood')] > pcutoff for version in ['new']}
hpAnkleR_mask = {version: data[version].loc[:, ('HindpawAnkleR', 'likelihood')] > pcutoff for version in ['new', 'old']}
hpToeL_mask = {version: data[version].loc[:, ('HindpawToeL', 'likelihood')] > pcutoff for version in ['new', 'old']}
hpKnuckleL_mask = {version: data[version].loc[:, ('HindpawKnuckleL', 'likelihood')] > pcutoff for version in ['new']}
hpAnkleL_mask = {version: data[version].loc[:, ('HindpawAnkleL', 'likelihood')] > pcutoff for version in ['new', 'old']}

# Define a colormap
colors = list(mcolors.TABLEAU_COLORS.values())

# Function to plot data for each limb
# Adjust how we access columns and apply masks
def plot_limb_data(limb_name, masks, data, color_idx):
    plt.figure(figsize=(10, 6))
    plt.title(f"{limb_name} Movement")

    # Define colormaps for gradients
    new_colormap = cm.Blues
    old_colormap = cm.Reds

    # Calculate the number of parts for color scaling
    num_parts_new = len(masks['new'])
    num_parts_old = len(masks['old'])

    # Plotting new data with blue gradient
    for i, (part, mask) in enumerate(masks['new'].items()):
        color_variant = new_colormap((i + 1) / num_parts_new)  # Gradual color
        plt.plot(data['new'].loc[mask, (part, 'y')].index,
                 data['new'].loc[mask, (part, 'y')].values,
                 label=f'New {part}', color=color_variant)
        plt.scatter(data['new'].loc[mask, (part, 'y')].index,
                    data['new'].loc[mask, (part, 'y')].values,
                    color=color_variant)

    # Plotting old data with red gradient
    for i, (part, mask) in enumerate(masks['old'].items()):
        color_variant = old_colormap((i + 1) / num_parts_old)  # Gradual color
        plt.plot(data['old'].loc[mask, (part, 'y')].index,
                 data['old'].loc[mask, (part, 'y')].values,
                 label=f'Old {part}', linestyle='dashed', color=color_variant)
        plt.scatter(data['old'].loc[mask, (part, 'y')].index,
                    data['old'].loc[mask, (part, 'y')].values,
                    color=color_variant)

    plt.xlabel('Frame')
    plt.ylabel('Y Position')
    plt.legend()

    plt.gca().invert_yaxis()


# Corrected function calls with actual masks
plot_limb_data(
    'Forepaw Left',
    {
        'new': {'ForepawToeL': fpToeL_mask['new'], 'ForepawKnuckleL': fpKnuckleL_mask['new'], 'ForepawAnkleL': fpAnkleL_mask['new']},
        'old': {'ForepawToeL': fpToeL_mask['old'], 'ForepawAnkleL': fpAnkleL_mask['old']}
    },
    data,
    color_idx=0
)

plot_limb_data(
    'Forepaw Right',
    {
        'new': {'ForepawToeR': fpToeR_mask['new'], 'ForepawKnuckleR': fpKnuckleR_mask['new'], 'ForepawAnkleR': fpAnkleR_mask['new']},
        'old': {'ForepawToeR': fpToeR_mask['old'], 'ForepawAnkleR': fpAnkleR_mask['old']}
    },
    data,
    color_idx=3
)

plot_limb_data(
    'Hindpaw Left',
    {
        'new': {'HindpawToeL': hpToeL_mask['new'], 'HindpawKnuckleL': hpKnuckleL_mask['new'], 'HindpawAnkleL': hpAnkleL_mask['new']},
        'old': {'HindpawToeL': hpToeL_mask['old'], 'HindpawAnkleL': hpAnkleL_mask['old']}
    },
    data,
    color_idx=6
)

plot_limb_data(
    'Hindpaw Right',
    {
        'new': {'HindpawToeR': hpToeR_mask['new'], 'HindpawKnuckleR': hpKnuckleR_mask['new'], 'HindpawAnkleR': hpAnkleR_mask['new']},
        'old': {'HindpawToeR': hpToeR_mask['old'], 'HindpawAnkleR': hpAnkleR_mask['old']}
    },
    data,
    color_idx=9
)

plt.show()