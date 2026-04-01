"""Validates run detection quality across experimental conditions."""
import os, glob, re
import pandas as pd
import helpers.utils as utils
from helpers.config import *

class CheckRuns:
    def __init__(self, file, mouseID, date, exp=None, speed=None,
                 repeat_extend=None, exp_wash=None, day=None,
                 vmt_type=None, vmt_level=None, prep=None):
        self.file, self.mouseID, self.date = file, mouseID, date
        (self.exp, self.speed, self.repeat_extend, self.exp_wash, self.day,
         self.vmt_type, self.vmt_level, self.prep) = (
            exp, speed, repeat_extend, exp_wash, day, vmt_type, vmt_level,
            prep)

        self.data = self.get_data()

    def get_data(self):
        # Get the data from the h5 file
        data = pd.read_hdf(self.file, key='real_world_coords')
        return data

    def find_sitting(self, data):
        pass

    def find_climbing(self, data):
        pass

    def check_runs(self):
        for r in self.data.index.get_level_values('Run').unique():
            run_data = self.data.loc[r]
            self.find_sitting(run_data)
            self.find_climbing(run_data)


class GetAllFiles:
    def __init__(self, directory=None, overwrite=False,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None):
        self.directory = directory
        self.overwrite = overwrite

        # store experiment conditions for error logging
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

    def GetFiles(self):
        files = utils.Utils().GetListofRunFiles(
            self.directory)  # gets dictionary of side, front and overhead
        # 3D files

        for j in range(0, len(files)):
            match = re.search(r'FAA-(\d+)', files[j])
            mouseID = match.group(1)
            pattern = "*%s*_Runs_Checked.h5" % mouseID
            dir = os.path.dirname(files[j])
            date = files[j].split(os.sep)[-1].split('_')[1]

            if not glob.glob(os.path.join(dir, pattern)) or self.overwrite:
                print(
                    f"########################################################"
                    f"\nFinding runs and extracting gait for {mouseID}..."
                    f"\n######################################################"
                )
                check_runs = CheckRuns(files[j], mouseID, date,
                                       exp=self.exp,
                                       speed=self.speed,
                                       repeat_extend=self.repeat_extend,
                                       exp_wash=self.exp_wash,
                                       day=self.day,
                                       vmt_type=self.vmt_type,
                                       vmt_level=self.vmt_level,
                                       prep=self.prep)
                try:
                    check_runs.check_runs()
                except Exception as e:
                    print(f"Error processing file {files[j]}: {str(e)}")
            else:
                print(f"Data for {mouseID} already exists. Skipping...")

        print(
            'All experiments have been mapped to real-world coordinates and '
            'saved.')


class GetConditionFiles:
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None,
                 vmt_level=None, prep=None, overwrite=False, save_frames=True):
        (self.exp, self.speed, self.repeat_extend, self.exp_wash, self.day,
         self.vmt_type, self.vmt_level, self.prep, self.overwrite,
         self.save_frames) = (
            exp, speed, repeat_extend, exp_wash, day, vmt_type, vmt_level,
            prep, overwrite, save_frames)

    def get_dirs(self):
        if self.speed:
            exp_speed_name = f"{self.exp}_{self.speed}"
        else:
            exp_speed_name = self.exp
        base_path = os.path.join(paths['filtereddata_folder'], exp_speed_name)

        # join any of the conditions that are not None in the order they
        # appear in the function as individual directories
        conditions = [self.repeat_extend, self.exp_wash, self.day,
                      self.vmt_type, self.vmt_level, self.prep]
        conditions = [c for c in conditions if c is not None]

        # if Repeats in conditions, add 'Wash' directory in the next
        # position in the list
        if 'Repeats' in conditions:
            idx = conditions.index('Repeats')
            conditions.insert(idx + 1, 'Wash')
        condition_path = os.path.join(base_path, *conditions)

        if os.path.exists(condition_path):
            print(f"Directory found: {condition_path}")
        else:
            raise FileNotFoundError(f"No path found {condition_path}")

        # Recursively find and process the final data directories
        self._process_subdirectories(condition_path)

    def _process_subdirectories(self, current_path):
        """
        Recursively process directories and get to the final data directories.
        """
        subdirs = [d for d in os.listdir(current_path) if
                   os.path.isdir(os.path.join(current_path, d))]

        # If subdirectories exist, traverse deeper
        if len(subdirs) > 0:
            print(f"Subdirectories found in {current_path}: {subdirs}")
            for subdir in subdirs:
                full_subdir_path = os.path.join(current_path, subdir)
                # Recursively process subdirectory
                self._process_subdirectories(full_subdir_path)
        else:
            # No more subdirectories, assume this is the final directory
            # with data
            print(f"Final directory: {current_path}")
            try:
                GetAllFiles(
                    directory=current_path,
                    overwrite=self.overwrite,
                    exp=self.exp,
                    speed=self.speed,
                    repeat_extend=self.repeat_extend,
                    exp_wash=self.exp_wash,
                    day=self.day,
                    vmt_type=self.vmt_type,
                    vmt_level=self.vmt_level,
                    prep=self.prep
                ).GetFiles()
            except Exception as e:
                print(f"Error processing directory {current_path}: {e}")


def main():
    # Get all data
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats',
                      exp_wash='Exp',
                      overwrite=False).get_dirs()  # should do all 3 days

    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended',
                      overwrite=False).get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended',
                      overwrite=False).get_dirs()
    GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended',
                      overwrite=False).get_dirs()


if __name__ == "__main__":
    main()
