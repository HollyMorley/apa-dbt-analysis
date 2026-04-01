import pandas as pd
import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from helpers.ConditionsFinder import BaseConditionFiles
from helpers import utils
from helpers.config import *

class OverviewRuns():
    def __init__(self, files, mouseIDs, dates,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None):
        self.files, self.mouseIDs, self.dates = files, mouseIDs, dates

        # store experiment conditions for error logging
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

        # If day is empty, look for 'Day(\d+)' in the first file
        if day is None:
            match = re.search(r'Day(\d+)', self.files[0]) if self.files else None
            if match:
                self.day = match.group(1)  # e.g. '1'
            else:
                self.day = None
        else:
            self.day = day

        # Depending on repeats vs. extended, load data differently
        # Define prep runs
        if self.speed and self.speed[:3] == 'Low':
            prep_runs = [0, 1]
        elif self.speed and self.speed[:4] == 'High':
            prep_runs = [0, 1, 2, 3, 4]
        else:
            raise ValueError(f"Unknown speed value: {self.speed}")

        if self.repeat_extend == 'Repeats':
            self.data = self.get_data(prep_runs)
        elif self.repeat_extend == 'Extended':
            self.data = self.get_data_extended(prep_runs)
        else:
            self.data = {}

    def remove_prep_runs(self, df, prep_runs):
        runs = df.index.get_level_values('Run').unique()
        runs_array = np.array(runs)
        valid_runs = runs_array[~np.isin(runs_array, prep_runs)]
        df = df.loc(axis=0)[valid_runs]
        # rename Run index shifted by prep_runs
        new_run_idx = valid_runs - len(prep_runs)
        df = df.rename(index=dict(zip(valid_runs, new_run_idx)), level='Run')
        return df

    def get_data(self,prep_runs):
        """
        Original: create dict of {mouseID: DataFrame}.
        """
        data = {}
        for idx, file in enumerate(self.files):
            df = pd.read_hdf(file, key='real_world_coords_runs')
            # Exclude prep runs
            df = self.remove_prep_runs(df, prep_runs)
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
            for f in mouse_files:
                match = re.search(r'Day(\d+)', f)
                if match:
                    day_num = int(match.group(1))
                else:
                    raise ValueError(f"Day number not found in file: {f}")

                day_num -= 1

                df = pd.read_hdf(f, key='real_world_coords_runs')
                # Exclude prep runs
                df = self.remove_prep_runs(df, prep_runs)
                mouse_data[day_num] = df
                days.append(day_num)

            # Concat and make 'Day' the top-level index
            data[mouse] = pd.concat(mouse_data, names=['Day'])
        return data

    def plot_runs(self):
        # Flatten all mice from config, so missing mice appear
        all_mice = [m for sublist in micestuff['mice_IDs'].values() for m in sublist]
        cmap = plt.get_cmap('tab20')
        num_colors = cmap.N

        # # Define prep runs
        # if self.speed and self.speed[:3] == 'Low':
        #     prep_runs = [0, 1]
        # elif self.speed and self.speed[:4] == 'High':
        #     prep_runs = [0, 1, 2, 3, 4]
        # else:
        #     raise ValueError(f"Unknown speed value: {self.speed}")

        # --------------------
        # REPEATS PLOT
        # --------------------
        if self.repeat_extend == 'Repeats':
            fig,ax = plt.subplots(1, 1, figsize=(10, 3))

            for midx, mouse in enumerate(all_mice):
                if mouse not in self.data:
                    continue
                runs = self.data[mouse].index.get_level_values('Run').unique()
                # Convert runs to a numpy array
                runs_array = np.array(runs)

                color = cmap(midx % num_colors)
                ax.scatter(runs_array + 0.5, [midx] * len(runs_array),
                            marker='s', color=color, label=mouse)

            for x in [0, 10, 30, 40]:
                ax.axvline(x=x, color='black', alpha=0.5, linestyle='--')

            ax.set(yticks=range(len(all_mice)), yticklabels=all_mice)
            ax.set_xlabel('Run (prep removed, consecutive)')
            ax.set_ylabel('Mouse ID')

            # Construct dynamic title
            conditions = [
                ('exp', self.exp),
                ('speed', self.speed),
                ('repeat_extend', self.repeat_extend),
                ('exp_wash', self.exp_wash),
                ('day', self.day),
                ('vmt_type', self.vmt_type),
                ('vmt_level', self.vmt_level),
                ('prep', self.prep)
            ]
            filtered = [value for (attr, value) in conditions if value is not None]
            ax.set_title(f"Runs for {'_'.join(filtered)}")

            plt.tight_layout()
            filename = self._make_filename(conditions)
            self.save_plots(fig, filename)

        # --------------------
        # EXTENDED PLOT
        # --------------------
        elif self.repeat_extend == 'Extended':
            fig,ax = plt.subplots(1, 1, figsize=(40, 3))

            for midx, mouse in enumerate(all_mice):
                if mouse not in self.data:
                    continue

                days_unique = self.data[mouse].index.get_level_values('Day').unique()
                for day_num in days_unique:
                    runs = self.data[mouse].loc[day_num].index.get_level_values('Run').unique()
                    runs_array = np.array(runs)

                    real_runs = runs_array + 40 * day_num

                    color = cmap(midx % num_colors)
                    ax.scatter(real_runs + 0.5, [midx] * len(real_runs),
                                marker='s', color=color, label=mouse if (midx == 0 and day_num == days_unique[0]) else "")

            for x in [0, 10, 110, 160]:
                ax.axvline(x=x, color='black', alpha=0.5, linestyle='--')

            ax.set(yticks=range(len(all_mice)), yticklabels=all_mice)
            ax.set_xlabel('Run (prep removed, consecutive across days)')
            ax.set_ylabel('Mouse ID')

            # Construct dynamic title
            conditions = [
                ('exp', self.exp),
                ('speed', self.speed),
                ('repeat_extend', self.repeat_extend),
                ('exp_wash', self.exp_wash),
                ('vmt_type', self.vmt_type),
                ('vmt_level', self.vmt_level),
                ('prep', self.prep)
            ]
            filtered = [value for (attr, value) in conditions if value is not None]
            ax.set_title(f"Extended Runs for {'_'.join(filtered)}")

            plt.tight_layout()
            filename = self._make_filename(conditions)
            self.save_plots(fig, filename)

        else:
            print(f"Unknown repeat_extend value: {self.repeat_extend}")

    def _make_filename(self, conditions):
        # conditions is a list of (attr, val)
        # Build something like: "exp_APAChar__speed_LowHigh__repeat_extend_Repeats.png"
        parts = []
        for (attr, val) in conditions:
            if val is not None:
                # remove spaces, parentheses, etc.
                safe_val = str(val).replace(' ', '_').replace('(', '_').replace(')', '_')
                parts.append(f"{attr}-{safe_val}")

        if not parts:
            return "RunChecks.png"
        base = "__".join(parts) + ".png"
        return base

    def save_plots(self, fig, filename=None):
        # Save the figure
        dest_dir = os.path.join(paths['plotting_destfolder'], 'RunChecks')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_path = os.path.join(dest_dir, filename)

        # Print out where we're saving
        #print(f"[DEBUG] About to save figure to: {dest_path}")

        fig.savefig(dest_path)
        #print(f"[DEBUG] Finished fig.savefig() call.")

        # # Check if it actually exists immediately after saving:
        # exists_now = os.path.exists(dest_path)
        # print(f"[DEBUG] File exists after save? {exists_now}")


class GetAllFiles():
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
            parent_dir = os.path.dirname(self.directory)
            if not os.path.isdir(parent_dir):
                print(f"Parent dir not found for extended condition: {parent_dir}")
                return

            # Gather .h5 from subdirs named 'Day...'
            all_files = []
            all_mouseIDs = []
            all_dates = []
            for subd in os.listdir(parent_dir):
                if not subd.lower().startswith('day'):
                    continue
                sub_path = os.path.join(parent_dir, subd)
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
                print(f"No day-based .h5 files found under {parent_dir}")
                return

            overview_runs = OverviewRuns(
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
            overview_runs.plot_runs()

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

            overview_runs = OverviewRuns(
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
            overview_runs.plot_runs()


class GetConditionFiles(BaseConditionFiles):
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None,
                 vmt_type=None, vmt_level=None, prep=None):
        super().__init__(
            exp=exp, speed=speed, repeat_extend=repeat_extend, exp_wash=exp_wash,
            day=day, vmt_type=vmt_type, vmt_level=vmt_level, prep=prep
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
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day2').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day3').get_dirs()

    # Extended
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended').get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended').get_dirs()
    GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended').get_dirs()

if __name__ == '__main__':
    main()
