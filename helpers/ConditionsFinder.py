"""Base class for recursively finding data directories matching experimental conditions."""
import os
from helpers.config import paths

class BaseConditionFiles:
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None,
                 overwrite=False, recursive=True):
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep
        self.overwrite = overwrite
        self.recursive = recursive

    def get_dirs(self):
        # Build base path from exp/speed
        if self.speed:
            exp_speed_name = f"{self.exp}_{self.speed}"
        else:
            exp_speed_name = self.exp

        base_path = os.path.join(paths['filtereddata_folder'], exp_speed_name)

        # Build the rest of the conditions
        conditions = [self.repeat_extend, self.exp_wash, self.day,
                      self.vmt_type, self.vmt_level, self.prep]
        conditions = [c for c in conditions if c is not None]
        if 'Repeats' in conditions:
            idx = conditions.index('Repeats')
            conditions.insert(idx + 1, 'Wash')

        condition_path = os.path.join(base_path, *conditions)

        if not os.path.exists(condition_path):
            raise FileNotFoundError(f"No path found {condition_path}")

        # Recursively process subdirectories from that path
        self._process_subdirectories(condition_path)

    def _process_subdirectories(self, current_path):
        print("Checking subdirectories in:", current_path)
        subdirs = [
            d for d in os.listdir(current_path)
            if os.path.isdir(os.path.join(current_path, d))
               and d.lower() != 'bin'
        ]

        if subdirs and self.recursive:
            for subdir in subdirs:
                self._process_subdirectories(os.path.join(current_path, subdir))
        else:
            # Final directory
            self.process_final_directory(current_path)

    def process_final_directory(self, directory):
        """
        Subclasses must implement this method to specify
        how we actually process the final directory.
        """
        raise NotImplementedError
