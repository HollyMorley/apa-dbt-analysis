import matplotlib.pyplot as plt
from curlyBrace import curlyBrace
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_rgb

def add_vertical_brace_curly(ax, y0, y1, x, xoffset, label=None, k_r=0.1, int_line_num=2, fontdict=None, rot_label=0, **kwargs):
    """
    Add a vertical curly brace using the curlyBrace package.
    The brace is drawn at the given x coordinate.
    """
    fig = ax.figure

    fontdict = fontdict or {}
    if 'fontsize' in kwargs:
        fontdict['fontsize'] = kwargs.pop('fontsize')

    p1 = [x, y0]
    p2 = [x, y1]
    # Do not pass the label here.12
    brace = curlyBrace(fig, ax, p1, p2, k_r=k_r, bool_auto=True, str_text=label,
                       int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)

def add_horizontal_brace_curly(ax, x0, x1, y, label=None, k_r=0.1, int_line_num=2, fontdict=None, **kwargs):
    """
    Add a horizontal curly brace using the curlyBrace package.
    The brace is drawn at the given y coordinate.
    """
    fig = ax.figure

    fontdict = fontdict or {}
    if 'fontsize' in kwargs:
        fontdict['fontsize'] = kwargs.pop('fontsize')

    # Swap p1 and p2 so that the brace opens toward the plot.
    p1 = [x1, y]
    p2 = [x0, y]
    brace = curlyBrace(fig, ax, p2, p1, k_r=k_r, bool_auto=True, str_text=label,
                       int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)


def add_cluster_brackets_heatmap(manual_clusters, feature_names, ax, horizontal=True, vertical=True,
                                 base_line_num = 2, label_offset=4, fs=6, distance_from_plot=-0.5):
    #### Add cluster boundaries ####
    cluster_names = {v: k for k, v in manual_clusters['cluster_values'].items()}

    # Compute cluster boundaries based on sorted order.
    cluster_boundaries = {}
    for idx, feat in enumerate(feature_names):
        cl = manual_clusters['cluster_mapping'].get(feat, -1)
        if cl not in cluster_boundaries:
            cluster_boundaries[cl] = {"start": idx, "end": idx}
        else:
            cluster_boundaries[cl]["end"] = idx
    # For each cluster, adjust boundaries by 0.5 (to align with cell edges).
    for i, (cl, bounds) in enumerate(cluster_boundaries.items()):
        # Define boundaries in data coordinates.
        x0, x1 = bounds["start"], bounds["end"]
        y0, y1 = bounds["start"], bounds["end"]

        k_r = 0.1
        span = abs(y1 - y0)
        desired_depth = 0.1  # or any value that gives you the uniform look you want
        k_r_adjusted = desired_depth / span if span != 0 else k_r

        # Alternate the int_line_num value for every other cluster:
        int_line_num = base_line_num + label_offset if i % 2 else base_line_num

        if vertical:
            # Add a vertical curly brace along the left side.
            add_vertical_brace_curly(ax, y0, y1, x=distance_from_plot, xoffset=1, label=cluster_names.get(cl, f"Cluster {cl}"),
                                        k_r=k_r_adjusted, int_line_num=int_line_num, fontsize=fs)
        if horizontal:
            # Add a horizontal curly brace along the top.
            add_horizontal_brace_curly(ax, x0, x1, y=distance_from_plot, label=cluster_names.get(cl, f"Cluster {cl}"),
                                          k_r=k_r_adjusted * -1, int_line_num=int_line_num, fontsize=fs)

# Helper function to darken a hex color
def darken_color(hex_color, factor=0.7):
    # factor < 1 will darken the color
    rgb = mcolors.to_rgb(hex_color)
    dark_rgb = tuple([x * factor for x in rgb])
    return mcolors.to_hex(dark_rgb)

# def get_colors(type):
#     if type == ['APA2','Wash2']:
#         color_1 = "#B11C73" #2FCAD0"
#         color_2 = "#589061"  # "#218EDC" #2F7AD0"
#         colors = (color_1, color_2)
#     elif type == ['APA1','APA2']:
#         color_1 = "#91e3e6"
#         color_2 = "#B11C73"
#         colors = (color_1, color_2)
#
#     return colors

def get_color_phase(phase):
    if phase == 'APA1':
        color = "#D87799"
    elif phase == 'APA2':
        color = "#B11C73" #2FCAD0"
    elif phase == 'Wash1':
        color = "#95CCD8"
    elif phase == 'Wash2':
        color = "#3E9BDD"  # "#218EDC" #2F7AD0"
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return color

def get_color_speedpair(speed):
    if speed == 'LowHigh':
        color = "#288733"
    elif speed == 'LowMid':
        color = "#95BD53"
    elif speed == 'HighLow':
        color = "#C44094" #2FCAD0"
    else:
        raise ValueError(f"Unknown speed: {speed}")
    return color

def get_color_stride(s):
    if s == 0:
        color = "#49080F"
    elif s == -1:
        color = "#810E1A"
    elif s == -2:
        color = "#DC182C"
    elif s == -3:
        color = "#ED5A68"
    else:
        raise ValueError(f"Unknown stride: {s}")
    return color

def get_color_pc(pc, colormap='gist_ncar', n_pcs=12, chosen_pcs=False):
    import matplotlib.colors as mcolors
    if not chosen_pcs:
        # length of possible PCs is 12, so we can use the first 12 colors from Set1
        if pc < 0 or pc >= n_pcs:
            raise ValueError(f"PC index {pc} is out of range. Must be between 0 and {n_pcs - 1}.")

        cm = plt.get_cmap(colormap)
        color = mcolors.to_hex(cm(pc / (n_pcs - 1)))  # Normalize pc to [0, 1]
    else:
        # If chosen_pcs is True, we use a predefined set of colors
        if pc == 1:
            color = "#04A2A2"
        elif pc == 3:
            color = "#5B5EA6"
        elif pc == 7:
            color = "#BD2459"
        elif pc == 5:
            color = "#330000"
        elif pc == 6:
            color = "#dd1188"
        elif pc == 8:
            color = "#ffaa33"
    return color


def get_ls_stride(s):
    if s == -1:
        ls = "-"
    elif s == -2:
        ls = "--"
    elif s == -3:
        ls = ":"
    elif s == 0:
        ls = "-"
    return ls

def get_line_style_mice(m):
    if m == '1035243':
        ls = 'solid'
    elif m == '1035244':
        ls = 'dotted'
    elif m == '1035245':
        ls = 'dashed'
    elif m == '1035246':
        ls = 'dashdot'
    elif m == '1035249':
        ls = (0, (1, 10)) # loosely dotted
    elif m == '1035250':
        ls = (5, (10, 3)) # long dash with offset
    elif m == '1035297':
        ls = (0, (5, 10)) # loosely dashed
    elif m == '1035298':
        ls = (0, (1, 1, 1, 1, 1, 1)) # densely dotted
    elif m == '1035299':
        ls = (0, (3, 1, 1, 1, 1, 1)) # densely dashdotdotted
    elif m == '1035301':
        ls = (0, (1, 1)) # densely dotted
    elif m == '1035302':
        ls = (0, (1, 1, 1, 1)) # densely dashdotted
    else:
        raise ValueError(f"Unknown mouse ID: {m}")
    return ls

def get_marker_style_mice(m):
    if m == '1035243':
        marker = 'o'
    elif m == '1035244':
        marker = '^'
    elif m == '1035245':
        marker = 's'
    elif m == '1035246':
        marker = 'D'
    elif m == '1035249':
        marker = 'x'
    elif m == '1035250':
        marker = '<'
    elif m == '1035297':
        marker = 'P'
    elif m == '1035298':
        marker = 'h'
    elif m == '1035299':
        marker = '>'
    elif m == '1035301':
        marker = 'v'
    elif m == '1035302':
        marker = '1'
    else:
        raise ValueError(f"Unknown mouse ID: {m}")
    return marker

def get_color_mice(m, cmap='YlGnBu', speedordered=None):
    import matplotlib.colors as mcolors
    if not speedordered:
        all_mice = ['1035243', '1035244', '1035245', '1035246',
                    '1035250', '1035297',
                    '1035299', '1035301'] # removed '1035249', '1035298', '1035302'
    else:
        all_mice = speedordered
    if m not in all_mice:
        raise ValueError(f"Unknown mouse ID: {m}")
    cm = plt.get_cmap(cmap)
    # Map to a narrower range (e.g. 0.3-0.8)
    lower, upper = 0.3, 0.8
    index = all_mice.index(m) / (len(all_mice) - 1)
    index_scaled = lower + index * (upper - lower)
    color = cm(index_scaled)
    return mcolors.to_hex(color)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))


def create_custom_colormap(lower_color, middle_color, upper_color, scaling=1.0, N=256):
    """
    Create a custom colormap using three specified colors, with a scaling parameter controlling
    the rate at which the colors change from the middle color to the extremes.

    Parameters:
    - lower_color: list or array-like, RGB values (normalized to 0-1) for the lower extreme.
    - middle_color: list or array-like, RGB values (normalized to 0-1) for the center.
    - upper_color: list or array-like, RGB values (normalized to 0-1) for the upper extreme.
    - scaling: float, controls the transition speed. Values > 1 produce a faster change from
               the middle to the extreme, while values < 1 yield a slower transition.
    - N: int, number of discrete color levels (default is 256).

    Returns:
    - cmap: a matplotlib ListedColormap instance.
    """
    colors = []

    for i in range(N):
        t = i / (N - 1)

        if t < 0.5:
            # For lower segment: calculate distance from the middle.
            u = (0.5 - t) / 0.5  # u=1 at t=0, u=0 at t=0.5.
            factor = 1 - (1 - u) ** scaling
            # Blend from middle_color (at u=0) down to lower_color (at u=1).
            color = np.array(middle_color) + factor * (np.array(lower_color) - np.array(middle_color))
        else:
            # For upper segment: calculate distance from the middle.
            u = (t - 0.5) / 0.5  # u=0 at t=0.5, u=1 at t=1.
            factor = 1 - (1 - u) ** scaling
            # Blend from middle_color (at u=0) up to upper_color (at u=1).
            color = np.array(middle_color) + factor * (np.array(upper_color) - np.array(middle_color))

        colors.append(color)

    return mcolors.ListedColormap(colors)

def gradient_colors(start_hex, end_hex, n):
    start_rgb = mcolors.to_rgb(start_hex)
    end_rgb = mcolors.to_rgb(end_hex)
    return [
        mcolors.to_hex(
            [start_rgb[i] + (end_rgb[i] - start_rgb[i]) * frac
             for i in range(3)]
        )
        for frac in np.linspace(0, 1, n)
    ]
def make_triple_cmap(base_color):
    # Light: mix with white (90% white)
    light_rgb = tuple([0.9 + 0.1 * c for c in mcolors.to_rgb(base_color)])
    mid_rgb = mcolors.to_rgb(base_color)
    dark_rgb = darken_color(base_color, factor=0.7)
    return mcolors.LinearSegmentedColormap.from_list('triple_cmap', [light_rgb, mid_rgb, dark_rgb])


def make_cmap_with_white_bottom(base_cmap, n_colors=256):
    # Sample the base colormap
    colors = base_cmap(np.linspace(0, 1, n_colors))
    # Replace the first color with white
    colors[0] = np.array([1, 1, 1, 1])  # RGBA white
    # Create new colormap
    new_cmap = mcolors.ListedColormap(colors)
    return new_cmap

def add_significance_stars(ax, positions_p1, positions_p2, data_p1, data_p2, pvals, height_buffer=0.1, fs=7): # currently for pca feature plots
    for i, p in enumerate(pvals):
        # Determine y position for the star: max of both groups + buffer
        max_y = max(max(data_p1[i]), max(data_p2[i]))
        y = max_y + height_buffer

        # Choose number of stars based on p-value
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            star = ''

        if star != '':
            # Draw line between bars
            x1, x2 = positions_p1[i], positions_p2[i]
            ax.plot([x1, x1, x2, x2], [y-0.05, y, y, y-0.05], lw=1, color='k')
            # Add text
            ax.text((x1 + x2) / 2, y + 0.02, star, ha='center', va='bottom', fontsize=fs)

def plot_phase_bars():
    apa1_color = get_color_phase('APA1')
    apa2_color = get_color_phase('APA2')
    wash1_color = get_color_phase('Wash1')
    wash2_color = get_color_phase('Wash2')

    boxy = 1
    height = 0.02
    patch1 = plt.axvspan(xmin=9.5, xmax=59.5, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
    patch2 = plt.axvspan(xmin=59.5, xmax=109.5, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
    patch3 = plt.axvspan(xmin=109.5, xmax=134.5, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
    patch4 = plt.axvspan(xmin=134.5, xmax=159.5, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
    patch1.set_clip_on(False)
    patch2.set_clip_on(False)
    patch3.set_clip_on(False)
    patch4.set_clip_on(False)


def make_phase_cmap(base_color, light=0.75, dark=0.6, name="phase_grad"):
    """
    Create a light-to-dark colormap from a base color.
    light: fraction blended toward white for the start color (0..1)
    dark:  fraction blended toward black for the end color (0..1)
    """
    base = np.array(to_rgb(base_color))

    def blend_to(target, t):
        return (1 - t) * base + t * np.array(target)

    light_rgb = blend_to((1, 1, 1), light)  # toward white
    dark_rgb  = blend_to((0, 0, 0), dark)   # toward black

    colors = np.vstack([light_rgb, base, dark_rgb])
    return LinearSegmentedColormap.from_list(name, colors, N=256)


