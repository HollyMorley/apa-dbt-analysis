"""Compiles per-mouse trial data into single per-condition pickle files."""
import pandas as pd
import re
import os
import pickle

from helpers.ConditionsFinder import BaseConditionFiles
from helpers import utils
from helpers.config import *

class PrepareSingleConditionFiles():
    def __init__(self, files, mouseIDs, dates, exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None):
        self.files, self.mouseIDs, self.dates = files, mouseIDs, dates

        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

        # Depending on repeats vs. extended, load data differently
        # Define prep runs
        if self.speed and self.speed[:3] == 'Low':
            prep_runs = [0, 1]
        elif self.speed and self.speed[:4] == 'High':
            prep_runs = [0, 1, 2, 3, 4]
        else:
            raise ValueError(f"Unknown speed value: {self.speed}")

        if self.repeat_extend == 'Repeats':
            self.data = self.get_data_repeats(prep_runs)
        elif self.repeat_extend == 'Extended':
            self.data = self.get_data_extended(prep_runs)
        else:
            raise ValueError(f"Unknown repeat_extend value, maybe you are trying VMT?")

    def get_data_repeats(self, prep_runs):
        data = {}
        for idx, file in enumerate(self.files):
            df = pd.read_hdf(file, key='real_world_coords_runs')
            # Exclude prep runs
            df = self.remove_prep_runs(df, prep_runs,days_offset=0)
            data[self.mouseIDs[idx]] = df
        return data

    def get_data_extended(self,prep_runs):
        """
        For extended experiments, gather day-based data
        into a multi-index: top-level 'Day'.
        """
        all_mice = [m for sublist in micestuff['mice_IDs'].values() for m in sublist]
        data = {}
        for mouse in all_mice:
            # Gather files for this mouse
            mouse_files = [f for f, m in zip(self.files, self.mouseIDs) if m == mouse]
            # Sort by Day(\d+)
            mouse_files = sorted(
                mouse_files,
                key=lambda x: int(re.search(r'Day(\d+)', x).group(1)) if re.search(r'Day(\d+)', x) else 0
            )
            if not mouse_files:
                continue

            days = []
            mouse_data = {}
            days_offset = 0
            frame_offset = 0

            existing_days = {}
            for f in mouse_files:
                match = re.search(r'Day(\d+)', f)
                if match:
                    day_num = int(match.group(1)) - 1
                    existing_days[day_num] = f

            max_day = max(existing_days.keys())

            for day_num in range(max_day + 1):

                if day_num not in existing_days:
                    print(f"Day {day_num} not found for mouse {mouse}.")
                    # account for the non standard trial per day
                    if mouse == '1035302' and self.speed == 'HighLow':
                        if day_num == 0:
                            days_offset += 20
                        elif day_num == 1:
                            days_offset += 50
                        elif day_num == 2:
                            days_offset += 50
                    else:
                        days_offset += 40

                    frame_offset += 1000  # or whatever typical spacing you use
                    continue

                f = existing_days[day_num]
                df = pd.read_hdf(f, key='real_world_coords_runs')
                # Exclude prep runs and adjust Run index using days_offset
                df = self.remove_prep_runs(df, prep_runs, days_offset)

                # Apply the cumulative frame offset to the FrameIdx index level
                df = df.rename(index=lambda x: x + frame_offset, level='FrameIdx')

                # Update frame_offset based on the maximum FrameIdx from the current day's data.
                # Adding 1 ensures the next day's FrameIdx starts after the current max.
                frame_offset = df.index.get_level_values('FrameIdx').max() + 1000

                mouse_data[day_num] = df

                # Update days_offset as you already do for runs
                if mouse == '1035302' and self.speed == 'HighLow':
                    if day_num == 0:
                        days_offset += 20
                    elif day_num == 1:
                        days_offset += 50
                    elif day_num == 2:
                        days_offset += 50
                else:
                    days_offset += 40

            # Concat and make 'Day' the top-level index
            data[mouse] = pd.concat(mouse_data, names=['Day'])
        return data

    def remove_prep_runs(self, df, prep_runs, days_offset):
        runs = df.index.get_level_values('Run').unique()
        runs_array = np.array(runs)
        valid_runs = runs_array[~np.isin(runs_array, prep_runs)]
        df = df.loc(axis=0)[valid_runs]
        # rename Run index shifted by prep_runs
        noprep_run_idx = valid_runs - len(prep_runs)
        # shift index according to days_offset
        day_run_idx = noprep_run_idx + days_offset
        df = df.rename(index=dict(zip(valid_runs, day_run_idx)), level='Run')
        return df

    def find_swing_stance_borders(self):
        for mouseID in self.data.keys():
            stance_continuous_mask = self.data[mouseID].xs('SwSt',level=1,axis=1) == locostuff['swst_vals_2025']['st']
            swing_continuous_mask = self.data[mouseID].xs('SwSt',level=1,axis=1) == locostuff['swst_vals_2025']['sw']

            stance_edges = pd.DataFrame(index=self.data[mouseID].index, columns=stance_continuous_mask.columns, data=np.nan)
            for limb in stance_edges.keys():
                stance_idx = stance_continuous_mask[limb][stance_continuous_mask[limb]].index.get_level_values('FrameIdx')
                swing_idx = swing_continuous_mask[limb][swing_continuous_mask[limb]].index.get_level_values('FrameIdx')
                stance_blocks = utils.Utils().find_blocks(stance_idx.values,gap_threshold=3,block_min_size=3)
                swing_blocks = utils.Utils().find_blocks(swing_idx.values,gap_threshold=3,block_min_size=3)
                stance_mask = stance_edges.index.get_level_values('FrameIdx').isin(stance_blocks[:,0])
                swing_mask = stance_edges.index.get_level_values('FrameIdx').isin(swing_blocks[:,0])
                stance_edges.loc[stance_mask, limb] = locostuff['swst_vals_2025']['st']
                stance_edges.loc[swing_mask, limb] = locostuff['swst_vals_2025']['sw']
            # add level 'SwSt_discrete' into columns
            stance_edges.columns = pd.MultiIndex.from_product([stance_edges.columns, ['SwSt_discrete']])
            # add stance_edges to self.data, under the same limbs but sublevel 'SwSt_discrete'
            self.data[mouseID] = pd.concat([self.data[mouseID],stance_edges],axis=1)

    def get_saved_data_filename(self):
        # Generate a consistent filename for saving/loading self.data
        if self.repeat_extend == 'Repeats':
            base_dir = os.path.dirname(self.files[0])
            filename = "allmice.pickle"
            return os.path.join(base_dir, filename)
        elif self.repeat_extend == 'Extended':
            # remove day from path
            base_dir = '\\'.join(os.path.dirname(self.files[0]).split('\\')[:-1])
            filename = "allmice.pickle"
            return os.path.join(base_dir, filename)

    def save_files(self):
        filepath = self.get_saved_data_filename()
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved data to: {filepath}")



class GetAllFiles():
    # Class-level set to track directories we've already processed
    processed_dirs = set()
    def __init__(self, directory=None,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None):
        self.directory = directory
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

    def get_files(self):
        """
        If 'Extended', gather .h5 from all Day subdirectories so we can plot them all at once.
        Otherwise, do the original logic for 'Repeats' or other cases.
        """
        if self.repeat_extend == 'Extended':
            #parent_dir = os.path.dirname(self.directory)
            if not os.path.isdir(self.directory):
                print(f"Parent dir not found for extended condition: {self.directory}")
                return

            # Gather .h5 from subdirs named 'Day...'
            all_files = []
            all_mouseIDs = []
            all_dates = []
            for subd in os.listdir(self.directory):
                if not subd.lower().startswith('day'):
                    continue
                sub_path = os.path.join(self.directory, subd)
                if os.path.isdir(sub_path):
                    these_files = utils.Utils().GetListofRunFiles(sub_path)
                    for f in these_files:
                        match = re.search(r'FAA-(\d+)', f)
                        if not match:
                            continue
                        all_files.append(f)
                        all_mouseIDs.append(match.group(1))
                        date_part = f.split(os.sep)[-1].split('_')[1]
                        all_dates.append(date_part)

            if not all_files:
                print(f"No day-based .h5 files found under {self.directory}")
                return

            print(f"##############################################################################################\n"
                  f"Found {len(all_files)} files for extended condition: {self.directory}.\n"
                  f"Processing... \n"
                  f"##############################################################################################")
            prepare_files = PrepareSingleConditionFiles(
                files=all_files,
                mouseIDs=all_mouseIDs,
                dates=all_dates,
                exp=self.exp,
                speed=self.speed,
                repeat_extend=self.repeat_extend,
                exp_wash=self.exp_wash,
                day=self.day,
                vmt_type=self.vmt_type,
                vmt_level=self.vmt_level,
                prep=self.prep
            )
            prepare_files.find_swing_stance_borders()
            prepare_files.save_files()

        else:
            # Original logic for Repeats
            files = utils.Utils().GetListofRunFiles(self.directory)
            if not files:
                print(f"No run files found in directory: {self.directory}")
                return

            mouseIDs = []
            dates = []
            for f in files:
                match = re.search(r'FAA-(\d+)', f)
                if match:
                    mouseIDs.append(match.group(1))
                else:
                    mouseIDs.append(None)
                try:
                    date = f.split(os.sep)[-1].split('_')[1]
                    dates.append(date)
                except IndexError:
                    dates.append(None)

            valid_indices = [
                i for i, (m, d) in enumerate(zip(mouseIDs, dates))
                if m is not None and d is not None
            ]
            filtered_files = [files[i] for i in valid_indices]
            filtered_mouseIDs = [mouseIDs[i] for i in valid_indices]
            filtered_dates = [dates[i] for i in valid_indices]

            if not filtered_files:
                print("No valid run files to process after filtering.")
                return

            print(f"##############################################################################################\n"
                    f"Found {len(filtered_files)} files for repeats condition: {self.directory}.\n"
                    f"Processing... \n"
                    f"##############################################################################################")

            prepare_files = PrepareSingleConditionFiles(
                files=filtered_files,
                mouseIDs=filtered_mouseIDs,
                dates=filtered_dates,
                exp=self.exp,
                speed=self.speed,
                repeat_extend=self.repeat_extend,
                exp_wash=self.exp_wash,
                day=self.day,
                vmt_type=self.vmt_type,
                vmt_level=self.vmt_level,
                prep=self.prep
            )
            prepare_files.find_swing_stance_borders()
            prepare_files.save_files()


class GetConditionFiles(BaseConditionFiles):
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None,
                 vmt_type=None, vmt_level=None, prep=None):
        if repeat_extend == 'Extended':
            recursive = False
        else:
            recursive = True
        super().__init__(
            exp=exp, speed=speed, repeat_extend=repeat_extend, exp_wash=exp_wash,
            day=day, vmt_type=vmt_type, vmt_level=vmt_level, prep=prep, recursive=recursive
        )

    def process_final_directory(self, directory):
        GetAllFiles(
            directory=directory,
            exp=self.exp,
            speed=self.speed,
            repeat_extend=self.repeat_extend,
            exp_wash=self.exp_wash,
            day=self.day,
            vmt_type=self.vmt_type,
            vmt_level=self.vmt_level,
            prep=self.prep,
        ).get_files()

def main():
    # Repeats
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1').get_dirs()
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day2').get_dirs()
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day3').get_dirs()

    # Extended
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended').get_dirs()
    GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended').get_dirs()

if __name__ == '__main__':
    main()