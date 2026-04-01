"""Renders static 3D skeleton images from pose data."""
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

from helpers.config import *

file = paths["filtereddata_folder"] + r"\APAChar_LowHigh\Extended\allmice.pickle"
savedir = PLOTS_ROOT + r"\Tracking"
mouse = '1035246'

frames = [188260, 188262, 188274, 188273, 188272, 188271, 358114, 358115, 358128, 358161, 358065, 358045, 358140, 358160]
bodyparts = micestuff['bodyparts']
skeleton = micestuff['skeleton']

with open(file, 'rb') as f:
    data = pickle.load(f)

mouse_data = data[mouse].loc(axis=0)[0]
## --------------- Ghost Skeleton 3D plot -------------------

# choose your frame and how many behind to show
ghost_frames = [358161, 188274]
n_prior = 40

for f in ghost_frames:
    frames_traj = list(range(f - n_prior, f + 1))

    # pull out those frames
    traj_stack = mouse_data.loc(axis=0)[:,:,frames_traj] \
                       .droplevel([0,1], axis=0)
    traj_body_parts = traj_stack.loc(axis=1)[bodyparts]

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111, projection='3d')

    # build a spring colormap spanning your trajectory
    cmap   = plt.cm.get_cmap('YlOrRd')
    colors = cmap(np.linspace(0, 1, len(frames_traj)))

    # plot the ghost skeletons in spring fade alpha if you like
    for i, fr in enumerate(frames_traj):
        X = traj_body_parts.loc[fr]
        xp = X.xs('x', level='coords')
        yp = X.xs('y', level='coords')
        zp = X.xs('z', level='coords')

        for j1, j2 in skeleton:
            ax.plot(
                [xp[j1], xp[j2]],
                [yp[j1], yp[j2]],
                [zp[j1], zp[j2]],
                c=colors[i],
                alpha=0.6,  # keep a bit of transparency
                lw=1
            )

    # over‑plot the current frame in solid spring‑end color
    Xc  = traj_body_parts.loc[f]
    xpc = Xc.xs('x', level='coords')
    ypc = Xc.xs('y', level='coords')
    zpc = Xc.xs('z', level='coords')
    last_color = colors[-1]
    for j1, j2 in skeleton:
        ax.plot(
            [xpc[j1], xpc[j2]],
            [ypc[j1], ypc[j2]],
            [zpc[j1], zpc[j2]],
            c=last_color,
            lw=2
        )
    # ax.scatter(xpc, ypc, zpc, c=last_color, s=30)

    # equal‐axis scaling
    max_range = np.array([
        xpc.max() - xpc.min(),
        ypc.max() - ypc.min(),
        zpc.max() - zpc.min()
    ]).max() / 2.0
    mid_x = (xpc.max() + xpc.min()) * 0.5
    mid_y = (ypc.max() + ypc.min()) * 0.5
    mid_z = (zpc.max() + zpc.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # clean background
    ax.set_facecolor('none')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    plt.show()

    views = [
        (90, -90, 'top'),  # straight down
        (0, -90, 'side'),  # from the side
        (0, 0, 'front'),  # from the front
        (18, -45, 'tilted')  # 30° up, 45° around
    ]

    for elev, azim, name in views:
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        fig.savefig(
            os.path.join(savedir, f'3D_ghost_{f}_{name}.png'),
            dpi=300
        )
        fig.savefig(
            os.path.join(savedir, f'3D_ghost_{f}_{name}.svg'),
            dpi=300
        )
    plt.close(fig)


## ---------------- Normal skeleton 3D plot ----------------

mouse_frames = mouse_data.loc(axis=0)[:,:,frames]
mouse_frames = mouse_frames.droplevel([0, 1], axis=0)

mouse_body_parts = mouse_frames.loc(axis=1)[bodyparts].copy()

for f in frames:
    X = mouse_body_parts.loc[f]

    # Separate coords via cross-section
    xp = X.xs('x', level='coords')
    yp = X.xs('y', level='coords')
    zp = X.xs('z', level='coords')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.get_cmap('summer')
    colors = cmap(np.linspace(0, 1, len(xp)))

    # scatter colored by body‑length
    ax.scatter(
        xp, yp, zp,
        c=colors,
        marker='o'
    )

    # Make the axes equal
    max_range = np.array([
        xp.max() - xp.min(),
        yp.max() - yp.min(),
        zp.max() - zp.min()
    ]).max() / 2.0
    mid_x = (xp.max() + xp.min()) * 0.5
    mid_y = (yp.max() + yp.min()) * 0.5
    mid_z = (zp.max() + zp.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # skeleton lines colored by the first joint’s position along the body
    name_to_idx = {name: i for i, name in enumerate(xp.index)}
    for joint1, joint2 in skeleton:
        idx = name_to_idx[joint1]
        ax.plot(
            [xp[joint1], xp[joint2]],
            [yp[joint1], yp[joint2]],
            [zp[joint1], zp[joint2]],
            '-', c=colors[idx]
        )

    ax.set_facecolor('none')  # no pane fill
    fig.patch.set_facecolor('white')  # or 'none' if you want transparency
    ax.grid(False)  # no grid lines

    # turn off the built‑in spines/ticks
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ——— draw custom X/Y/Z axes lines (or arrows) behind your data ———
    # pick a length
    # length of each axis line
    L = 50
    # Tail12 x coordinate
    tail12_x = xp['Tail12']
    tail12_y = 0

    # — X axis: run +X from the mouse’s tail‑end y position —
    ax.plot(
        [tail12_x, tail12_x + L],  # x from tail12_x → tail12_x+L
        [tail12_y, tail12_y],  # y fixed at tail12_y
        [0, 0],  # z fixed at 0
        '-', lw=1.5
    )
    ax.quiver(
        tail12_x, tail12_y, 0,  # start at (tail12_x, tail12_y, 0)
        L, 0, 0,  # point +X
        arrow_length_ratio=0.05, lw=1.5
    )
    ax.text(
        tail12_x + L, tail12_y, 0, 'X', fontsize=12
    )

    # — Y axis: still at x=tail12_x, z=0, origin y=50, pointing –Y —
    y_origin = 0
    ax.plot(
        [tail12_x, tail12_x],  # x fixed
        [y_origin, y_origin + L],  # y from 50 → 50−L
        [0, 0],  # z fixed
        '-', lw=1.5
    )
    ax.quiver(
        tail12_x, y_origin, 0,  # start at (tail12_x,50,0)
        0, L, 0,  # point –Y
        arrow_length_ratio=0.05, lw=1.5
    )
    ax.text(
        tail12_x, y_origin + L, 0, 'Y', fontsize=12
    )

    # — Z axis: run +Z from the mouse’s tail‑end y position —
    ax.plot(
        [tail12_x, tail12_x],  # x fixed at tail12_x
        [tail12_y, tail12_y],  # y fixed at tail12_y
        [0, L],  # z from 0 → L
        '-', lw=1.5
    )
    ax.quiver(
        tail12_x, tail12_y, 0,  # start at (tail12_x, tail12_y, 0)
        0, 0, L,  # point +Z
        arrow_length_ratio=0.05, lw=1.5
    )
    ax.text(
        tail12_x, tail12_y, L, 'Z', fontsize=12
    )

    views = [
        (90, -90, 'top'),  # straight down
        (0, -90, 'side'),  # from the side
        (0, 0, 'front'),  # from the front

        (18, -43, 'tilted')  # 30° up, 45° around
    ]


    for elev, azim, name in views:
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        fig.savefig(
            os.path.join(savedir, f'3D_{f}_{name}.png'),
            dpi=300
        )
        fig.savefig(
            os.path.join(savedir, f'3D_{f}_{name}.svg'),
            dpi=300
        )
    plt.close(fig)


