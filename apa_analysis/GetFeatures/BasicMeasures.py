"""Orchestrate stride-level and run-level kinematic measure extraction across experimental conditions."""
import pandas as pd
import os
import pickle
import warnings
from tqdm import tqdm

from multiprocessing import Pool


from helpers import utils
from helpers.config import *

from helpers.ConditionsFinder import BaseConditionFiles
from apa_analysis.GetFeatures.MeasuresByStride import CalculateMeasuresByStride, RunMeasures
from apa_analysis.GetFeatures.MeasuresByRun import CalculateMeasuresByRun

class Save():
    def __init__(self, file, exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None, buffer_size=0.25,
                 analyses=["behaviour", "single", "multi"]):
        self.file = file

        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep
        self.buffer_size = buffer_size
        self.analyses = analyses

        self.XYZw = self.get_data()
        self.CalculateMeasuresByStride = CalculateMeasuresByStride

        self.error_logs = []

    def get_data(self):
        with open(self.file[0], 'rb') as handle:
            XYZw = pickle.load(handle)
        data = XYZw
        return data

    def find_pre_post_transition_strides(self, mouseID, r, numstrides=3):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        post_frame = self.XYZw[mouseID].loc(axis=0)[r, 'Transition'].loc(axis=1)[['ForepawL','ForepawR']].iloc[0].name
        stepping_limb = self.XYZw[mouseID].loc(axis=0)[r,:,post_frame].loc(axis=1)['initiating_limb'].values
        if len(stepping_limb) > 1:
            raise ValueError("More than one stepping limb found")
        elif len(stepping_limb) == 0:
            raise ValueError("No stepping limb found")
        else:
            stepping_limb = stepping_limb[0]

        stance_mask_pre = self.XYZw[mouseID].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
        swing_mask_pre = self.XYZw[mouseID].loc(axis=0)[r, ['RunStart']].loc(axis=1)[stepping_limb, 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']
        stance_mask_post = self.XYZw[mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
        swing_mask_post = self.XYZw[mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']

        stance_idx_pre = pd.DataFrame(self.XYZw[mouseID].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'SwSt_discrete'][stance_mask_pre].tail(numstrides))
        swing_idx_pre = pd.DataFrame(self.XYZw[mouseID].loc(axis=0)[r,['RunStart']].loc(axis=1)[stepping_limb,'SwSt_discrete'][swing_mask_pre].tail(numstrides))
        stance_idx_post = pd.DataFrame(self.XYZw[mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'SwSt_discrete'][stance_mask_post].head(2))
        swing_idx_post = pd.DataFrame(self.XYZw[mouseID].loc(axis=0)[r, ['Transition','RunEnd']].loc(axis=1)[stepping_limb, 'SwSt_discrete'][swing_mask_post].head(2))

        stance_idx_pre['Stride_no'] = np.sort(np.arange(1,len(stance_idx_pre)+1)*-1)
        swing_idx_pre['Stride_no'] = np.sort(np.arange(1,len(swing_idx_pre)+1)*-1)
        stance_idx_post['Stride_no'] = np.arange(0,len(stance_idx_post))
        swing_idx_post['Stride_no'] = np.arange(0,len(swing_idx_post))


        # Combine pre and post DataFrames
        combined_df = pd.concat([stance_idx_pre,swing_idx_pre, stance_idx_post, swing_idx_post]).sort_index(level='FrameIdx')

        return combined_df

    def find_pre_post_transition_strides_ALL_RUNS(self, mouseID):
        #view = 'Side'
        SwSt = []

        # # If there is a 'Day' level, iterate over that:
        # day_levels = (
        #     self.XYZw[mouseID].index.get_level_values('Day').unique()
        #     if 'Day' in self.XYZw[mouseID].index.names
        #     else [None]
        # )
        # todo drop the 'Day' level if it exists - MAY WANT TO RETAIN THIS LONG TERM
        if 'Day' in self.XYZw[mouseID].index.names:
            self.XYZw[mouseID].reset_index('Day', drop=True, inplace=True)

        for r in self.XYZw[mouseID].index.get_level_values(level='Run').unique().astype(int):
            try:
                stsw = self.find_pre_post_transition_strides(mouseID=mouseID,r=r)
                SwSt.append(stsw)
            except:
                pass
                #print(f'Cant get stsw for run {r}, mouse {mouseID}')
        SwSt_df = pd.concat(SwSt)

        return SwSt_df

    def get_measures_byrun_bystride(self, SwSt, mouseID):
        warnings.simplefilter(action='ignore', category=FutureWarning)

        st_mask = (SwSt.loc(axis=1)[['ForepawR', 'ForepawL'],'SwSt_discrete'] == locostuff['swst_vals_2025']['st']).any(axis=1)
        stride_borders = SwSt[st_mask]

        temp_single_list = []
        temp_multi_list = []
        temp_run_list = []

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
        conditions = dict(conditions)

        for r in tqdm(stride_borders.index.get_level_values(level='Run').unique(), desc=f"Processing: {mouseID}"):
            stepping_limb = np.array(['ForepawR','ForepawL'])[(stride_borders.loc(axis=0)[r].loc(axis=1)[['ForepawR','ForepawL']].count() > 1).values][0]

            # If either single or multi kinematics is selected, do the stride-by-stride measures:
            if "single" in self.analyses or "multi" in self.analyses:
                for sidx, s in enumerate(stride_borders.loc(axis=0)[r].loc(axis=1)['Stride_no']):#[:-1]):
                    # if len(stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')) <= sidx + 1:
                    #     print("Can't calculate s: %s" %s)
                    try:
                        stride_start = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx]
                        stride_end = stride_borders.loc(axis=0)[r].index.get_level_values(level='FrameIdx')[sidx + 1] - 1 # todo check i am right to consider the previous frame the end frame

                        #class_instance = self.CalculateMeasuresByStride(self.XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
                        calc_obj = CalculateMeasuresByStride(self.XYZw, mouseID, r, stride_start, stride_end, stepping_limb, conditions)
                        measures_dict = measures_list(buffer=self.buffer_size)

                        runXstride_measures = RunMeasures(measures_dict, calc_obj, buffer_size=self.buffer_size, stride=s)

                        # Conditionally collect results:
                        if "single" in self.analyses:
                            single_val, _ = runXstride_measures.get_all_results()
                            temp_single_list.append(single_val)
                        if "multi" in self.analyses:
                            _, multi_val = runXstride_measures.get_all_results()
                            temp_multi_list.append(multi_val)
                    except Exception as e:
                        # Replace print with logging to error_logs
                        error_entry = {
                            'MouseID': mouseID,
                            'Run': r,
                            'Stride': s,
                            'Condition_exp': self.exp,
                            'Condition_speed': self.speed,
                            'Condition_repeat_extend': self.repeat_extend,
                            'Condition_exp_wash': self.exp_wash,
                            'Condition_day': self.day,
                            'Condition_vmt_type': self.vmt_type,
                            'Condition_vmt_level': self.vmt_level,
                            'Condition_prep': self.prep,
                            'Error': str(e)
                        }
                        self.error_logs.append(error_entry)

            # If behaviour measures are selected, run the run-level measures:
            if "behaviour" in self.analyses:
                try:
                    run_measures = CalculateMeasuresByRun(XYZw=self.XYZw, mouseID=mouseID, r=r,
                                                          stepping_limb=stepping_limb, conditions=conditions)
                    run_val = run_measures.run()
                    temp_run_list.append(run_val)
                except Exception as e:
                    error_entry = {
                        'MouseID': mouseID,
                        'Run': r,
                        'Stride': None,
                        'Condition_exp': self.exp,
                        'Condition_speed': self.speed,
                        'Condition_repeat_extend': self.repeat_extend,
                        'Condition_exp_wash': self.exp_wash,
                        'Condition_day': self.day,
                        'Condition_vmt_type': self.vmt_type,
                        'Condition_vmt_level': self.vmt_level,
                        'Condition_prep': self.prep,
                        'Error': str(e)
                    }
                    self.error_logs.append(error_entry)

        # Now concatenate results if computed, otherwise return empty DataFrames:
        measures_bystride_single = pd.concat(temp_single_list) if temp_single_list else pd.DataFrame()
        measures_bystride_multi = pd.concat(temp_multi_list) if temp_multi_list else pd.DataFrame()
        measures_byrun = pd.concat(temp_run_list) if temp_run_list else pd.DataFrame()

        return measures_bystride_single, measures_bystride_multi, measures_byrun

    def process_mouse_data(self, mouseID):
        try:
            # Process data for the given mouseID
            SwSt = self.find_pre_post_transition_strides_ALL_RUNS(mouseID=mouseID)
            self.mirror_coordinates(SwSt=SwSt, mouseID=mouseID)
            single_byStride, multi_byStride, byRun = self.get_measures_byrun_bystride(SwSt=SwSt, mouseID=mouseID)

            # add mouseID to SwSt index
            index = SwSt.index
            multi_idx_tuples = [(mouseID, a[0], a[1], a[2]) for a in index]
            multi_idx = pd.MultiIndex.from_tuples(multi_idx_tuples, names=['MouseID', 'Run', 'Stride', 'FrameIdx'])
            SwSt.set_index(multi_idx, inplace=True)

            return single_byStride, multi_byStride, byRun, SwSt
        except Exception as e:
            # Replace print with logging to error_logs
            error_entry = {
                'MouseID': mouseID,
                'Run': None,
                'Stride': None,
                'Condition_exp': self.exp,
                'Condition_speed': self.speed,
                'Condition_repeat_extend': self.repeat_extend,
                'Condition_exp_wash': self.exp_wash,
                'Condition_day': self.day,
                'Condition_vmt_type': self.vmt_type,
                'Condition_vmt_level': self.vmt_level,
                'Condition_prep': self.prep,
                'Error': f"Processing mouse error: {str(e)}"
            }
            self.error_logs.append(error_entry)
            return None, None, None, None
        finally:
            self.error_logs = []  # Reset after processing

    def mirror_coordinates(self, SwSt, mouseID):
        stepping_limb_runs = {}
        for r in SwSt.index.get_level_values('Run').unique():
            limb = SwSt.loc(axis=0)[r].xs('SwSt_discrete', level=1, axis=1).isna().any().index[
            SwSt.loc(axis=0)[r].xs('SwSt_discrete', level=1, axis=1).isna().any()]
            if len(limb) > 1:
                raise ValueError("More than one stepping limb found")
            else:
                stepping_limb_runs[r] = limb[0]
        stepping_limb_runs_df = pd.DataFrame.from_dict(stepping_limb_runs, orient='index', columns=['SteppingLimb'])

        L_mask = np.array(stepping_limb_runs_df.values == 'ForepawL').flatten()
        L_runs = stepping_limb_runs_df.index[L_mask]

        y_midline = structural_stuff['belt_width'] / 2
        data = self.XYZw[mouseID].copy(deep=True)

        # Create a mask for the rows to update (runs where stepping limb is 'ForepawL')
        left_mask = self.XYZw[mouseID].index.get_level_values('Run').isin(L_runs)

        # Create a slice for selecting all 'y' columns for all body parts
        cols_y = pd.IndexSlice[:, 'y']

        # Calculate mirrored coordinates in one go (vectorized)
        mirrored_coords = 2 * y_midline - data.loc[left_mask, cols_y]

        # Update the data DataFrame
        data.loc[left_mask, cols_y] = mirrored_coords

        # Save the mirrored data back to self.XYZw for the mouseID
        self.XYZw[mouseID] = data

    def process_mouse_data_wrapper(self, args):
        mouseID = args
        single_byStride, multi_byStride, byRun, SwSt = self.process_mouse_data(mouseID)
        return single_byStride, multi_byStride, byRun, SwSt, self.error_logs  # Return error_logs

    def save_all_measures_parallel(self):
        #pool = Pool(cpu_count())
        pool = Pool(processes=6) # was 4
        # Initialize multiprocessing Pool with number of CPU cores
        results = []

        # Process data for each mouseID in parallel
        for mouseID in self.XYZw.keys():
            print(f"{mouseID} Processing...")
            result = pool.apply_async(self.process_mouse_data_wrapper, args=((mouseID),))
            results.append(result)

        print(results)

        # Aggregate results
        single_byStride_all, multi_byStride_all, byRun_all, SwSt_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        all_error_logs = []  # Initialize a list to collect all error logs

        for result in results:
            single_byStride, multi_byStride, byRun, SwSt, error_logs = result.get()
            if single_byStride is not None:
                single_byStride_all = pd.concat([single_byStride_all, single_byStride])
            if multi_byStride is not None:
                multi_byStride_all = pd.concat([multi_byStride_all, multi_byStride])
            if byRun is not None:
                byRun_all = pd.concat([byRun_all, byRun])
            if SwSt is not None:
                SwSt_all = pd.concat([SwSt_all, SwSt])
            if error_logs:
                all_error_logs.extend(error_logs)  # Collect error logs

        # Convert error logs to DataFrame and save to CSV
        if all_error_logs:
            error_df = pd.DataFrame(all_error_logs)
            error_df.to_csv(os.path.join(dir, "Error_Logs.csv"), index=False)

        single_byStride_all = single_byStride_all.apply(pd.to_numeric, errors='coerce', downcast='float')
        multi_byStride_all = multi_byStride_all.apply(pd.to_numeric, errors='coerce', downcast='float')
        byRun_all = byRun_all.apply(pd.to_numeric, errors='coerce', downcast='float')
        #SwSt_all = SwSt_all.apply(pd.to_numeric, errors='coerce', downcast='float')

        # Write to HDF files
        # if 'Day' not in con:
        #     dir = os.path.join(paths['filtereddata_folder'], con)
        # else:
        #     dir = utils.Utils().Get_processed_data_locations(con)
        dir = os.path.dirname(self.file[0])

        if "single" in self.analyses and not single_byStride_all.empty:
            single_byStride_all.to_hdf(os.path.join(dir, "MEASURES_single_kinematics_runXstride.h5"),
                                       key='single_kinematics', mode='w')
        if "multi" in self.analyses and not multi_byStride_all.empty:
            multi_byStride_all.to_hdf(os.path.join(dir, "MEASURES_multi_kinematics_runXstride.h5"),
                                      key='multi_kinematics', mode='w')
        if "behaviour" in self.analyses and not byRun_all.empty:
            byRun_all.to_hdf(os.path.join(dir, "MEASURES_behaviour_run.h5"),
                             key='behaviour', mode='w')
        SwSt_all.to_hdf(os.path.join(dir, "MEASURES_StrideInfo.h5"), key='stride_info', mode='w')

        # Wait for all processes to complete and collect results
        pool.close()
        pool.join()


class GetAllFiles():
    def __init__(self, directory=None,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None, analyses=["behaviour", "single", "multi"]):
        self.directory = directory
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep
        self.analyses = analyses

    def get_files(self):
        # Original logic for Repeats
        file = utils.Utils().GetAllMiceFiles(self.directory)
        if not file:
            print(f"No run file found in directory: {self.directory}")
            return

        save = Save(
            file=file,
            exp=self.exp,
            speed=self.speed,
            repeat_extend=self.repeat_extend,
            exp_wash=self.exp_wash,
            day=self.day,
            vmt_type=self.vmt_type,
            vmt_level=self.vmt_level,
            prep=self.prep,
            analyses=self.analyses,
        )
        save.save_all_measures_parallel()

class GetConditionFiles(BaseConditionFiles):
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None,
                 vmt_type=None, vmt_level=None, prep=None, analyses=["behaviour", "single", "multi"]):
        if repeat_extend == 'Extended':
            recursive = False
        else:
            recursive = True
        super().__init__(
            exp=exp, speed=speed, repeat_extend=repeat_extend, exp_wash=exp_wash,
            day=day, vmt_type=vmt_type, vmt_level=vmt_level, prep=prep, recursive=recursive
        )
        self.analyses = analyses

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
            analyses=self.analyses
        ).get_files()

def main():
    # Extended
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended', analyses=["single"]).get_dirs() # todo redo these with all analyses!!!!
    GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended', analyses=["single"]).get_dirs()
    GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended',  analyses=["single"]).get_dirs()

    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended', analyses=["behaviour", "single", "multi"]).get_dirs() # todo redo these with all analyses!!!!
    # GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended', analyses=["behaviour", "single", "multi"]).get_dirs()
    # GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended',  analyses=["behaviour", "single", "multi"]).get_dirs()
    #
    # # Repeats
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1').get_dirs()
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day2').get_dirs()
    # GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day3').get_dirs()

if __name__ == '__main__':
    main()
