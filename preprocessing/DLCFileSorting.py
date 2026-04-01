"""Sorts and copies DLC analysis output files into condition-organized folders."""
import pandas as pd
import os
import shutil
import re

from helpers.config import *

# Specify directories for each camera type
dirs = {
    'side': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round3",
    'front': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round2",
    'overhead': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round2"
}
dlc_dest = paths['filtereddata_folder']#r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round5_Dec24"

MouseIDs = micestuff['mice_IDs']

# Function to create directories if they don't exist
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory: {path}')


# Function to find and copy files recursively based on exp_cats and MouseIDs
def copy_files_recursive(dest_dir, current_dict, current_path, MouseIDs, overwrite):
    if isinstance(current_dict, dict):
        # Check if we are at the level where 'A' and 'B' keys are present
        if 'A' in current_dict and 'B' in current_dict:
            # Process for both 'A' and 'B' groups
            for mouse_group in ['A', 'B']:
                if mouse_group in MouseIDs:
                    for date in current_dict[mouse_group]:
                        try:
                            print(f'Processing: {current_path}, {mouse_group}, {date}')
                            copy_files_for_mouse_group(dest_dir, current_path, mouse_group, date, MouseIDs,
                                                       overwrite)
                        except Exception as e:
                            print(f"Error copying {current_path}, {mouse_group}, {date}: {e}")
                           # raise  # or comment out if you want to continue

        else:
            # Continue recursion if we haven't reached 'A'/'B' level yet
            for key, value in current_dict.items():
                new_path = os.path.join(current_path, key)
                copy_files_recursive(dest_dir, value, new_path, MouseIDs, overwrite)
    else:
        # If current_dict is not a dict, something is wrong, as we should not reach here
        raise ValueError("Unexpected structure in experiment categories.")


# Function to find and copy files for a specific mouse group and date, stitching if needed
def copy_files_for_mouse_group(dest_dir, current_path, mouse_group, date, MouseIDs, overwrite):
    final_dest_dir = os.path.join(dest_dir, current_path)  # Keep experiment structure, but no A/B in folder path
    ensure_dir_exists(final_dest_dir)

    current_src_dir = {
        'side': os.path.join(dirs['side'], date),
        'front': os.path.join(dirs['front'], date),
        'overhead': os.path.join(dirs['overhead'], date),
    }
    current_timestamp_dir = os.path.join(paths['video_folder'], date)

    file_found = False

    # ------------------------------------------------------------------
    # 1) RENAME 10352450 --> 1035250, but ONLY if date == '20230325'
    #    (unchanged from your current code)
    # ------------------------------------------------------------------
    if date == '20230325':
        for cam, cam_dir_path in current_src_dir.items():
            for old_name in os.listdir(cam_dir_path):
                if "10352450" in old_name:
                    old_path = os.path.join(cam_dir_path, old_name)
                    new_name = old_name.replace("10352450", "1035250")
                    new_path = os.path.join(cam_dir_path, new_name)
                    if not os.path.exists(new_path):
                        print(f"Renaming {old_path} to {new_path}")
                        os.rename(old_path, new_path)

    # ------------------------------------------------------------------
    # 2) CHECK FOR ANY UNKNOWN MOUSE IDs (for ALL dates)
    # ------------------------------------------------------------------
    all_known_ids = set(MouseIDs['A'] + MouseIDs['B'])
    for cam, cam_dir_path in current_src_dir.items():
        for fname in os.listdir(cam_dir_path):
            if fname.endswith('.h5'):
                match = re.search(r"FAA-(\d+)", fname)
                if match:
                    found_id = match.group(1)
                    if found_id not in all_known_ids:
                        print(f"[WARNING] File {fname} in {cam_dir_path} has unknown mouse ID = {found_id}")

    # ------------------------------------------------------------------
    # 3) Process each mouse in the chosen group
    # ------------------------------------------------------------------
    for mouse_id in MouseIDs[mouse_group]:
        try:
            # Gather all .h5 data files and .csv timestamp files for this mouse/date
            relevant_files = []
            timestamp_relevant_files = []
            for cam in ['side', 'front', 'overhead']:
                cam_folder = current_src_dir[cam]
                # data
                relevant_files.extend([
                    f for f in os.listdir(cam_folder)
                    if cam in f and f.endswith('.h5')
                    and f"FAA-{mouse_id}" in f
                    and os.path.isfile(os.path.join(cam_folder, f))
                ])
                # timestamps
                if os.path.exists(current_timestamp_dir):
                    timestamp_relevant_files.extend([
                        f for f in os.listdir(current_timestamp_dir)
                        if cam in f and f.endswith('Timestamps.csv')
                        and f"FAA-{mouse_id}" in f
                        and os.path.isfile(os.path.join(current_timestamp_dir, f))
                    ])

            relevant_files.sort() # This will ensure '' comes first, followed by '_2', '_3'
            timestamp_relevant_files.sort()

            if not relevant_files:
                print(f"No relevant h5 files found for {mouse_id} on {date}")
                continue

            # Quick check: number of .h5 files vs timestamps
            if len(relevant_files) != len(timestamp_relevant_files):
                # your special exception for 1035249 / 20230306
                if mouse_id == '1035249' and date == '20230306':
                    timestamp_relevant_files = timestamp_relevant_files[3:]
                else:
                    raise ValueError(f"Error: Different # of timestamp files vs data files for {mouse_id} on {date}")

            # ------------------------------------------------------------------
            # If we have more than 3 data files total, you do multi-file stitching
            # else if == 3 data files total, it's "one side, one front, one overhead".
            # The difference is how we group them by camera & possibly stitch them.
            # Then we do a length check before saving to .h5.
            # ------------------------------------------------------------------
            if len(relevant_files) > 3:
                # We'll group by camera, read, stitch, fix mismatch, then save
                stitched_dfs = {'side': None, 'front': None, 'overhead': None}
                stitched_timestamps = {'side': None, 'front': None, 'overhead': None}

                for cam in stitched_dfs.keys():
                    # which files belong to this camera?
                    cam_files = [file_name for file_name in relevant_files if cam in file_name]
                    # order the files in the correct order by moving the last file to the first position
                    cam_files.insert(0, cam_files.pop())

                    # which timestamps belong to this camera?
                    cam_ts_files = [file_name for file_name in timestamp_relevant_files if cam in file_name]
                    cam_ts_files.insert(0, cam_ts_files.pop())

                    # read + stitch them
                    cam_dfs = []
                    for f in cam_files:
                        df = pd.read_hdf(os.path.join(current_src_dir[cam], f))
                        cam_dfs.append(df)
                    stitched_cam_df = pd.concat(cam_dfs, ignore_index=True)

                    # read + stitch timestamps
                    ts_dfs = []
                    for f in cam_ts_files:
                        df_ts = pd.read_csv(os.path.join(current_timestamp_dir, f))
                        ts_dfs.append(df_ts)
                    stitched_cam_ts = pd.concat(ts_dfs, ignore_index=True)

                    stitched_dfs[cam] = stitched_cam_df
                    stitched_timestamps[cam] = stitched_cam_ts

                # 3a. Fix row mismatch if needed (the known 1035249/20230326 case)
                if mouse_id == '1035249' and date == '20230326':
                    # Suppose front is correct, side & overhead are each 1 row too long
                    side_len      = len(stitched_dfs['side'])
                    front_len     = len(stitched_dfs['front'])
                    overhead_len  = len(stitched_dfs['overhead'])
                    if side_len == front_len + 1 and overhead_len == front_len + 1:
                        print(f"[INFO] Trimming last frame from side & overhead for {mouse_id} on {date}")
                        stitched_dfs['side']      = stitched_dfs['side'].iloc[:-1]
                        stitched_dfs['overhead']  = stitched_dfs['overhead'].iloc[:-1]
                        # if timestamps also mismatch, do the same:
                        if (len(stitched_timestamps['side']) == len(stitched_timestamps['front']) + 1
                            and len(stitched_timestamps['overhead']) == len(stitched_timestamps['front']) + 1):
                            stitched_timestamps['side'] = stitched_timestamps['side'].iloc[:-1]
                            stitched_timestamps['overhead'] = stitched_timestamps['overhead'].iloc[:-1]

                elif mouse_id == '1035245' and date == '20230325':
                    side_len = len(stitched_dfs['side'])
                    overhead_len = len(stitched_dfs['overhead'])
                    if overhead_len == side_len + 1:
                        print(f"[INFO] Trimming last frame from overhead for {mouse_id} on {date}")
                        stitched_dfs['overhead'] = stitched_dfs['overhead'].iloc[:-1]
                        # if timestamps also mismatch:
                        oh_ts_len = len(stitched_timestamps['overhead'])
                        side_ts_len = len(stitched_timestamps['side'])
                        if oh_ts_len == side_ts_len + 1:
                            stitched_timestamps['overhead'] = stitched_timestamps['overhead'].iloc[:-1]

                # 3b. Final check: all 3 cameras same length?
                side_len     = len(stitched_dfs['side'])
                front_len    = len(stitched_dfs['front'])
                overhead_len = len(stitched_dfs['overhead'])
                if not (side_len == front_len == overhead_len):
                    raise ValueError(f"Error: Mismatch rows for {mouse_id} on {date} after stitching/fix. "
                                     f"Lens= side:{side_len}, front:{front_len}, overhead:{overhead_len}")

                # 3c. Save to .h5 & CSV using the *first* filename for each camera (or whichever naming you prefer)
                for cam in ['side','front','overhead']:
                    cam_files = [file_name for file_name in relevant_files if cam in file_name]
                    cam_files.insert(0, cam_files.pop())
                    if not cam_files:
                        continue
                    # We'll use the first cam file name to save final result:
                    dest_file = os.path.join(final_dest_dir, cam_files[0])
                    if overwrite or not os.path.exists(dest_file):
                        print(f"Stitching and saving {cam_files} to {dest_file}")
                        stitched_dfs[cam].to_hdf(dest_file, key='df_with_missing', mode='w')

                    # same for timestamps
                    cam_ts_files = [file_name for file_name in timestamp_relevant_files if cam in file_name]
                    cam_ts_files.insert(0, cam_ts_files.pop())
                    if cam_ts_files:
                        dest_ts_file = os.path.join(final_dest_dir, cam_ts_files[0])
                        if overwrite or not os.path.exists(dest_ts_file):
                            print(f"Stitching and saving {cam_ts_files} to {dest_ts_file}")
                            stitched_timestamps[cam].to_csv(dest_ts_file, index=False)

                file_found = True

            elif len(relevant_files) == 3:
                # Exactly 3 data files total => side, front, overhead (one each).
                # We'll read them all, fix row mismatch, then save as new .h5.
                # And likewise handle timestamps identically.

                # Grab filenames for each camera
                side_file      = next(f for f in relevant_files if 'side' in f)
                front_file     = next(f for f in relevant_files if 'front' in f)
                overhead_file  = next(f for f in relevant_files if 'overhead' in f)
                side_df     = pd.read_hdf(os.path.join(current_src_dir['side'], side_file))
                front_df    = pd.read_hdf(os.path.join(current_src_dir['front'], front_file))
                overhead_df = pd.read_hdf(os.path.join(current_src_dir['overhead'], overhead_file))

                # 1) Read the single timestamp files for each camera:
                side_ts_file = next(f for f in timestamp_relevant_files if 'side' in f)
                front_ts_file = next(f for f in timestamp_relevant_files if 'front' in f)
                overhead_ts_file = next(f for f in timestamp_relevant_files if 'overhead' in f)
                side_ts = pd.read_csv(os.path.join(current_timestamp_dir, side_ts_file))
                front_ts = pd.read_csv(os.path.join(current_timestamp_dir, front_ts_file))
                overhead_ts = pd.read_csv(os.path.join(current_timestamp_dir, overhead_ts_file))

                # 2) Fix known mismatch if needed, trimming both data and timestamps:
                if mouse_id == '1035249' and date == '20230326':
                    if len(side_df) == len(front_df) + 1 and len(overhead_df) == len(front_df) + 1:
                        print(f"[INFO] Trimming last frame from side & overhead for {mouse_id} on {date}")
                        side_df = side_df.iloc[:-1]
                        overhead_df = overhead_df.iloc[:-1]
                        side_ts = side_ts.iloc[:-1]
                        overhead_ts = overhead_ts.iloc[:-1]

                elif mouse_id == '1035245' and date == '20230325':
                    if len(overhead_df) == len(side_df) + 1:
                        print(f"[INFO] Trimming last frame from overhead for {mouse_id} on {date}")
                        overhead_df = overhead_df.iloc[:-1]
                        #overhead_ts = overhead_ts.iloc[:-1] # todo the timestamps are not different len to other cams - THIS IS WEIRD?!?!

                # Final check
                if not (len(side_df) == len(front_df) == len(overhead_df)):
                    raise ValueError(f"Error: Mismatch rows for {mouse_id} on {date}"
                                     f"Lens= side:{len(side_df)}, front:{len(front_df)}, overhead:{len(overhead_df)}")

                if not (len(side_ts) == len(front_ts) == len(overhead_ts)):
                    raise ValueError(f"Error: Mismatch timestamps for {mouse_id} on {date}")

                # Save data + timestamps in one loop:
                for (cam, df, ts, file_name, ts_name) in [
                    ('side', side_df, side_ts, side_file, side_ts_file),
                    ('front', front_df, front_ts, front_file, front_ts_file),
                    ('overhead', overhead_df, overhead_ts, overhead_file, overhead_ts_file),
                ]:
                    data_dest = os.path.join(final_dest_dir, file_name)
                    ts_dest = os.path.join(final_dest_dir, ts_name)

                    # Save the data
                    if overwrite or not os.path.exists(data_dest):
                        print(f"[SINGLE-FILE-SAVE] {cam} data → {data_dest}")
                        df.to_hdf(data_dest, key='df_with_missing', mode='w')
                    else:
                        print(f"Skipped saving {cam} data (already exists) → {data_dest}")

                    # Save the timestamps
                    if overwrite or not os.path.exists(ts_dest):
                        print(f"[SINGLE-FILE-SAVE] {cam} timestamps → {ts_dest}")
                        ts.to_csv(ts_dest, index=False)
                    else:
                        print(f"Skipped saving {cam} timestamps (already exists) → {ts_dest}")
            else:
                print(f"Unexpected number of files found for {mouse_id} on {date}")

        except Exception as e:
            print(f"[WARNING] Skipping mouse {mouse_id} on {date} due to error:\n   {e}")
            # This ensures we continue to the next mouse_id in the loop
            continue

    # End of for mouse_id in MouseIDs[mouse_group]:

    if not file_found:
        print(f"No files found for {mouse_group} on {date}")



def manual_changes():
    # Define the paths for destination (where modifications occur) and source (where new files are copied from)
    highlow_dir = os.path.join(dlc_dest, 'APAChar_HighLow', 'Extended')

    # Define mouse ID for the changes
    mouse_id = '1035243'

    # Define manual date modifications
    changes = [
        {'action': 'delete', 'mouse_id': mouse_id, 'date': '20230325', 'day': 'Day1'},
        {'action': 'delete', 'mouse_id': mouse_id, 'date': '20230326', 'day': 'Day2'},
        {'action': 'move', 'mouse_id': mouse_id, 'date': '20230327', 'src_day': 'Day3', 'dst_day': 'Day1'},
        {'action': 'move', 'mouse_id': mouse_id, 'date': '20230328', 'src_day': 'Day4', 'dst_day': 'Day2'},
        {'action': 'copy', 'mouse_id': mouse_id, 'date': '20230329', 'src_day': '20230329', 'dst_day': 'Day3'},
        {'action': 'copy', 'mouse_id': mouse_id, 'date': '20230330', 'src_day': '20230330', 'dst_day': 'Day4'}
    ]

    for change in changes:
        if change['action'] == 'delete':
            # Delete relevant files (both data and timestamps) from the destination day
            day_path = os.path.join(highlow_dir, change['day'])
            for file in os.listdir(day_path):
                if f"FAA-{change['mouse_id']}" in file and f"{change['date']}" in file:
                    file_path = os.path.join(day_path, file)
                    if os.path.exists(file_path):
                        print(f"Deleting {file_path}")
                        os.remove(file_path)


        elif change['action'] == 'move':
            # Move files from one destination day to another within dlc_dest, including timestamps
            src_day_path = os.path.join(highlow_dir, change['src_day'])
            dst_day_path = os.path.join(highlow_dir, change['dst_day'])
            for file in os.listdir(src_day_path):
                if f"FAA-{change['mouse_id']}" in file and f"{change['date']}" in file:
                    src_file = os.path.join(src_day_path, file)
                    dst_file = os.path.join(dst_day_path, file)
                    print(f"Moving {src_file} to {dst_file}")
                    shutil.move(src_file, dst_file)

        elif change['action'] == 'copy':
            for cam, source_dir_cam in dirs.items():
                # Copy files from dlc_dir (source) to the appropriate destination day
                src_day = os.path.join(source_dir_cam, change['src_day'])  # Source is dlc_dir with the date
                dst_day = os.path.join(highlow_dir, change['dst_day'])  # Destination is within dlc_dest

                # Ensure destination directory exists
                ensure_dir_exists(dst_day)

                for file in os.listdir(src_day):
                    if f"FAA-{change['mouse_id']}" in file and f"{change['date']}" in file and cam in file:
                        src_file = os.path.join(src_day, file)
                        dst_file = os.path.join(dst_day, file)
                        print(f"Copying {src_file} to {dst_file}")
                        shutil.copy(src_file, dst_file)

                # also copy timestamps
                src_ts = os.path.join(paths['video_folder'], change['src_day'])
                dst_ts = dst_day
                for file in os.listdir(src_ts):
                    if f"FAA-{change['mouse_id']}" in file and f"{change['date']}" in file and cam in file and 'Timestamps' in file:
                        src_file = os.path.join(src_ts, file)
                        dst_file = os.path.join(dst_ts, file)
                        print(f"Copying {src_file} to {dst_file}")
                        shutil.copy(src_file, dst_file)



def final_checks():
    # Check each camera's directory separately
    excluded_folders = {}

    for cam, cam_dir in dirs.items():
        all_folders = os.listdir(cam_dir)

        # Flatten the exp_cats dictionary to get all the dates
        included_dates = []
        for category, subcats in exp_cats.items():
            for subcat, phases in subcats.items():
                if isinstance(phases, dict):
                    for phase, days in phases.items():
                        if isinstance(days, dict):
                            for day, mice_dates in days.items():
                                for mouse_group, date_list in mice_dates.items():
                                    included_dates.extend(date_list)
                        elif isinstance(days, list):
                            # e.g., included_dates.extend(days)
                            pass
                elif isinstance(phases, list):
                    # This is where you'd handle PerceptionTest if subcat='A' or 'B'
                    included_dates.extend(phases)

        # Exclude manually handled days
        manual_dates = ['20230329', '20230330']

        # Check which folders are not included in exp_cats or manually handled
        excluded_folders[cam] = [
            folder for folder in all_folders if folder not in included_dates and folder not in manual_dates
        ]

    # Print the excluded folders for each camera
    for cam, folders in excluded_folders.items():
        if folders:
            print(f"{cam.capitalize()} camera: The following folders are not included in the experiment categories or manual changes: {folders}")
        else:
            print(f"All {cam} camera folders are accounted for in exp_cats or manual changes.")


# Example usage
overwrite = False  # Set this to True if you want to overwrite files
copy_files_recursive(dlc_dest, exp_cats, '', MouseIDs, overwrite)

# Perform manual changes
manual_changes()

# Final check for any folders left out
final_checks()
