# Some useful helper functions for video analysis
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
from helpers.config import *
from matplotlib.patches import Polygon
import sys
import numpy as np
import re
import math
import pandas as pd
import os
import pickle

class Utils:
    def __init__(self):
        super().__init__()

    def generate_path(self, *args, **kwargs):
        path_parts = [str(arg) for arg in args if arg]
        for key, value in kwargs.items():
            if value:
                path_parts.append(key)
                path_parts.append(value)
        return os.path.join(*path_parts)

    def GetDFs(self, conditions, reindexed_loco=False):
        '''
        :param conditions: list of experimental conditions want to plot/analyse eg 'APAChar_HighLow', 'APAChar_LowHigh_Day1', 'APAVMT_LowHighac'. NB make sure to include the day if for a condition which has repeats
        :return: dictionary holding all dataframes held under the requested conditions
        '''
        if reindexed_loco:
            file_suffix = 'Runs__IdxCorr'
        else:
            file_suffix = 'Runs'
        print('Conditions to be loaded:\n%s' % conditions)
        data = dict.fromkeys(conditions)
        for conidx, con in enumerate(conditions):
            if 'Day' not in con:
                files = self.GetlistofH5files(directory=r"%s\%s" % (paths['filtereddata_folder'], con), filtered=True,
                                                       suffix=file_suffix)
            else:
                splitcon = con.split('_')
                conname = "_".join(splitcon[0:2])
                dayname = splitcon[-1]
                w = splitcon[-2]
                if 'Repeats' in con:
                    files = self.GetlistofH5files(
                        directory=r"%s\%s\Repeats\%s\%s" % (paths['filtereddata_folder'], conname, w, dayname), filtered=True,
                        suffix=file_suffix)
                elif 'Extended' in con:
                    files = self.GetlistofH5files(
                        directory=r"%s\%s\Extended\%s" % (paths['filtereddata_folder'], conname, dayname), filtered=True,
                        suffix=file_suffix)
                else:
                    files = self.GetlistofH5files(
                        directory=r"%s\%s\%s" % (paths['filtereddata_folder'], conname, dayname), filtered=True,
                        suffix=file_suffix)
            mouseIDALL = list()
            dateALL = list()

            # sort lists of filenames so that in same order for reading
            files['Side'].sort()
            files['Front'].sort()
            files['Overhead'].sort()

            for f in range(0, len(files['Side'])):
                mouseID = os.path.basename(files['Side'][f]).split('_')[3]
                mouseIDALL.append(mouseID)
                date = os.path.basename(files['Side'][f]).split('_')[1]
                dateALL.append(date)

            # data['%s' % con] = dict.fromkeys(dateALL)
            data['%s' % con] = dict.fromkeys(mouseIDALL)
            for n, name in enumerate(mouseIDALL):
                # data['%s' % con]['%s' %name] = dict.fromkeys(['Side', 'Front', 'Overhead'])
                data['%s' % con]['%s' % name] = {
                    'Date': dateALL[n],
                    'Side': pd.read_hdf(files['Side'][n]),
                    'Front': pd.read_hdf(files['Front'][n]),
                    'Overhead': pd.read_hdf(files['Overhead'][n])
                }
        return data

    def Get_vid_loc_from_analysed_file(self, file, view):
        day = os.path.basename(file).split('_')[1]
        video_base_path = "\\".join([paths['video_folder'], day])
        new_filename =  os.path.basename(file).replace(vidstuff['scorers'][view],'').replace('.h5', '.avi')
        video_path = os.path.join(video_base_path, new_filename)
        return video_path

    def Get_timestamps_from_analyse_file(self, file, view):
        day = os.path.basename(file).split('_')[1]
        video_base_path = "\\".join([paths['video_folder'], day])
        new_filename =  os.path.basename(file).replace(vidstuff['scorers'][view],'').replace('.h5', '_Timestamps.csv')
        video_path = os.path.join(video_base_path, new_filename)
        return video_path


    def Get_processed_data_locations(self, con):
        splitcon = con.split('_')
        conname = "_".join(splitcon[0:2])
        dayname = splitcon[-1]
        w = splitcon[-2]
        if 'Repeats' in con:
            directory = r"%s\%s\Repeats\%s\%s" % (paths['filtereddata_folder'], conname, w, dayname)
        elif 'Extended' in con:
            directory = r"%s\%s\Extended\%s" % (paths['filtereddata_folder'], conname, dayname)
        else:
            directory = r"%s\%s\%s" % (paths['filtereddata_folder'], conname, dayname)
        return directory

    def Get_XYZw_DFs(self, conditions):
        print('Conditions to be loaded:\n%s' % conditions)
        data = dict.fromkeys(conditions)
        for conidx, con in enumerate(conditions):
            if 'Day' not in con:
                files = r"%s\%s" % (paths['filtereddata_folder'], con)
            else:
                directory = self.Get_processed_data_locations(con)
                file = "%s\\allmice_%s_XYZw.pickle" % (directory,con)
            with open(file, 'rb') as handle:
                XYZw = pickle.load(handle)
            data['%s' % con] = XYZw
        return data

    def Getlistofvideofiles(self, view, directory, filetype=".avi"):
        # function to get a list of video files in a directory. Current use is for bulk DLC analysis of videos with different models in each directory
        # view can be either 'side', 'overhead' or 'front'
        #vidfiles = glob("%s\\*%s%s" % (directory, scorer, filetype))
        ignore = ["labeled", "test"]
        vidfiles = [f for f in glob("%s/*%s*%s" % (directory, view, filetype)) if not any(j in f for j in ignore)]
        print(vidfiles)
        return vidfiles

    def GetListofMappedFiles(self, directory):
        mappedfiles = glob("%s\\*_mapped3D.h5" % directory)
        mappedfiles.sort()
        if not mappedfiles:
            print("No mapped files found in this directory")
        else:
            print("Files to be analysed are:\n"
                  "%d files\n"
                  "%s" % (len(mappedfiles), mappedfiles))
            return mappedfiles

    def GetListofRunFiles(self, directory):
        runfiles = glob("%s\\*_Runs.h5" % directory)
        runfiles.sort()
        if not runfiles:
            print("No run files found in this directory")
        else:
            print("Files to be analysed are:\n"
                  "%d files\n"
                  "%s" % (len(runfiles), runfiles))
            return runfiles

    def GetAllMiceFiles(self, directory):
        allmicefiles = glob("%s\\allmice.pickle" % directory)
        if not allmicefiles:
            print("No allmice files found in this directory")
        else:
            print("Files to be analysed are:\n"
                  "%d files\n"
                  "%s" % (len(allmicefiles), allmicefiles))
            return allmicefiles

    def GetlistofH5files(self, files=None, directory=None):
        datafiles_side = None
        datafiles_front = None
        datafiles_overhead = None

        if directory is not None and files is None:
            datafiles_side = glob("%s\\*%s*.h5" % (directory, 'side')) # !!!!!!!!!!!!GOT RID OF scorer_side!!!!!!!!!!!!!!!!!!!!!!!!!!
            datafiles_front = glob("%s\\*%s*.h5" % (directory, 'front'))
            datafiles_overhead = glob("%s\\*%s*.h5" % (directory, 'overhead'))

            datafiles_side.sort()
            datafiles_front.sort()
            datafiles_overhead.sort()

        elif files is not None and directory is None:
            datafiles_side = []
            datafiles_front = []
            datafiles_overhead = []
            for i in range(0, len(files)):
                if 'front' in files[i]:
                    front = files[i]
                    datafiles_front.append(front)
                elif 'overhead' in files[i]:
                    overhead = files[i]
                    datafiles_overhead.append(overhead)
                elif 'side' in files[i]:
                    side = files[i]
                    datafiles_side.append(side)

        if bool(datafiles_side) or bool(datafiles_front) or bool(datafiles_overhead):
            print("Files to be analysed are:\n"
                  "Side: %d files\n"
                  "%s\n"
                  "Front: %d files\n"
                  "%s\n"
                  "Overhead: %d files\n"
                  "%s" % (
                  len(datafiles_side), datafiles_side, len(datafiles_front), datafiles_front, len(datafiles_overhead),
                  datafiles_overhead))


            datafiles = {
                'Side': datafiles_side,
                'Front': datafiles_front,
                'Overhead': datafiles_overhead
            }
            return datafiles
        else:
            raise Exception("No files found, check file format.\nHint: you should be using the .h5 files\nHint: If specifying just one file, put this is list format still")

    def checkFilenamesMouseID(self, files):
        # Checks if mouse ID corresponds to correct mouse name
        if type(files) is dict:
            files = sorted({x for v in files.values() for x in v})

        for m in range(0, len(micestuff['mice_ID'])):
            mousefiles = [s for s in files if micestuff['mice_ID'][m] in s]
            match = [f for f in mousefiles if micestuff['mice_ID'][m] in f]
            if mousefiles != match:
                mislabeled = set(mousefiles) ^ set(match)
                print('The following file is labeled incorrectly:\n%s' % mislabeled)
                print('Code will now quit. Please correct this error and re-try!')
                sys.exit()
            else:
                print('All videos labeled correctly')

    def getFilepaths(self, data):
        filenameALL = list()
        skelfilenameALL = list()
        pathALL = list()
        for df in range(0, len(data)):
            filename = Path(data[df]).stem
            skelfilename = "%s_skeleton" %filename
            path = str(Path(data[df]).parent)
            filenameALL.append(filename)
            skelfilenameALL.append(skelfilename)
            pathALL.append(path)
        return filenameALL, skelfilenameALL, pathALL

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def get_cmap(self, n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    #def pxtocm(self, dataframe):

    def get_exp_details(self,file):
        exp = []
        runPhases = []
        splitrunPhases = []
        indexList = []
        splitindexList = []
        Acspeed = [] # actual speed cm/s
        Pdspeed = [] # perceived speed cm/s
        condition = []
        AcBaselineSpeed = []
        AcVMTSpeed = []
        AcWashoutSpeed = []
        PdBaselineSpeed = []
        PdVMTSpeed = []
        PdWashoutSpeed = []
        VMTcon = []
        VMTtype = []
        pltlabel = []

        if '20201130' in file:
            exp = 'APACharBaseline'
            runPhases = [list(range(0, 20))]
            indexList = ['BaselineRuns']
            Acspeed = 0
            Pdspeed = 0
            condition = 'Control'
        elif '20201201' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            Acspeed = 8
            Pdspeed = 8
            condition = 'FastToSlow'
        elif '20201202' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            Acspeed = 4
            Pdspeed = 4
            condition = 'SlowToFast'
        elif '20201203' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            condition = 'SlowToFast'
            Acspeed = 16
            Pdspeed = 16
        elif '20201204' in file:
            exp = 'APACharNoWash'
            runPhases = [list(range(0, 5)), list(range(5, 25))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25))]
            indexList = ['BaselineRuns', 'APARuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half']
            condition = 'FastToSlow'
            Acspeed = 4
            Pdspeed = 4
        elif '20201207' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'SlowToFast'
            Acspeed = 8
            Pdspeed = 8
        elif '20201208' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'FastToSlow'
            Acspeed = 16
            Pdspeed = 16
        elif '20201209' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'FastToSlow'
            Acspeed = 32
            Pdspeed = 32
        elif '20201210' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'SlowToFast'
            Acspeed = 32
            Pdspeed = 32
        elif '20201211' in file:
            exp = 'APAChar'
            runPhases = [list(range(0, 5)), list(range(5, 25)), list(range(25, 30))]
            splitrunPhases = [list(range(0, 5)), list(range(5, 15)), list(range(15, 25)), list(range(25, 30))]
            indexList = ['BaselineRuns', 'APARuns', 'WashoutRuns']
            splitindexList = ['BaselineRuns', 'APARuns 1st Half', 'APARuns 2nd Half', 'WashoutRuns']
            condition = 'PerceptionTest'
            Acspeed = 16
            Pdspeed = 100
        elif '20201214' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            AcBaselineSpeed = 4
            AcVMTSpeed = 4
            AcWashoutSpeed = 4
            PdBaselineSpeed = 4
            PdVMTSpeed = 16
            PdWashoutSpeed = 4
            VMTcon = 'Slow'
            VMTtype = 'Perceived change'
            pltlabel = 'Actual = 4cm/s, Perceived = 16cm/s'
        elif '20201215' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 32
            AcVMTSpeed = 32
            AcWashoutSpeed = 32
            PdBaselineSpeed = 32
            PdVMTSpeed = 4
            PdWashoutSpeed = 32
        elif '20201216' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 16
            AcVMTSpeed = 16
            AcWashoutSpeed = 16
            PdBaselineSpeed = 16
            PdVMTSpeed = 4
            PdWashoutSpeed = 16
            VMTcon = 'Fast'
            VMTtype = 'Perceived change'
            pltlabel = 'Actual = 16cm/s, Perceived = 4cm/s'
        elif '20201217' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 4
            AcVMTSpeed = 16
            AcWashoutSpeed = 4
            PdBaselineSpeed = 4
            PdVMTSpeed = 4
            PdWashoutSpeed = 4
            VMTcon = 'Slow'
            VMTtype = 'Actual change'
            pltlabel = 'Actual = 16cm/s, Perceived = 4cm/s'
        elif '20201218' in file:
            exp = 'VisuoMotTransf'
            runPhases = [list(range(0, 10)), list(range(10, 25)), list(range(25, 35))]
            indexList = ['BaselineRuns', 'VMTRuns', 'WashoutRuns']
            condition = 'SlowToFast'
            AcBaselineSpeed = 16
            AcVMTSpeed = 4
            AcWashoutSpeed = 16
            PdBaselineSpeed = 16
            PdVMTSpeed = 16
            PdWashoutSpeed = 16
            VMTcon = 'Fast'
            VMTtype = 'Actual change'
            pltlabel = 'Actual = 4cm/s, Perceived = 16cm/s'
        else:
            print('Somethings gone wrong, cannot find this file')

        details = {
            'exp': exp,
            'runPhases': runPhases,
            'splitrunPhases': splitrunPhases,
            'indexList': indexList,
            'splitindexList': splitindexList,
            'Acspeed': Acspeed,
            'Pdspeed': Pdspeed,
            'condition': condition,
            'AcBaselineSpeed': AcBaselineSpeed,
            'AcVMTSpeed': AcVMTSpeed,
            'AcWashoutSpeed': AcWashoutSpeed,
            'PdBaselineSpeed': PdBaselineSpeed,
            'PdVMTSpeed': PdVMTSpeed,
            'PdWashoutSpeed': PdWashoutSpeed,
            'VMT condition': VMTcon,
            'VMT type': VMTtype,
            'plt label': VMTtype
        }

        return details

    # def flatten_dict_keys(self, d, parent_key='', sep='_'):
    #     items = []
    #     for k, v in d.items():
    #         new_key = parent_key + sep + k if parent_key else k
    #         if isinstance(v, dict):
    #             items.extend(self.flatten_dict_keys(v, new_key, sep=sep).items())
    #         else:
    #             items.append((new_key, v))
    #     return dict(items)
    def flatten_dict_keys(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict_keys(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, v))
            else:
                items.append((new_key, [v]))
        return dict(items)

    def findConfidentMeans(self, data, label, coor):
        mask = data.loc(axis=1)[label, 'likelihood'] > pcutoff
        mean = np.mean(data.loc(axis=1)[label, coor][mask])
        return mean


    def getSpeedConditions(self, con, speed=list(expstuff['speeds'].keys())):
        matches = re.findall('|'.join(speed), con)
        if len(matches) == 2:
            if con.index(matches[0]) < con.index(matches[1]):
                speed_order = (matches[0], matches[1])
            else:
                speed_order = (matches[1], matches[0])
        else:
            speed_order = None

        return speed_order

    def sigmoid(self, x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    def frametotime(self, frame):
        time = (frame / fps) / 60
        sec, min = math.modf(time)
        sec = sec * 60
        print('%s mins, %s secs' % (min, sec))

    def combinetwohdfs(self):
        scorers = [vidstuff['scorers']['side'], vidstuff['scorers']['front'], vidstuff['scorers']['overhead']]
        for v, view in enumerate(vidstuff['cams']):
            df1 = pd.read_hdf(
                r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\20230317\bin\HM_20230317_APACharExt_FAA-1035244_L_%s_1%s.h5" % (
                view, scorers[v]))
            df2 = pd.read_hdf(
                r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\20230317\bin\HM_20230317_APACharExt_FAA-1035244_L_2_%s_1%s.h5" % (
                view, scorers[v]))
            newdf = pd.concat([df1, df2], ignore_index=True)
            newdf.to_hdf(
                r"M:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\20230317\HM_20230317_APACharExt_FAA-1035244_L_%s_1%s.h5" % (
                view, scorers[v]), key='view', mode='w')

    def frametotime_2vid(self, frame, extra):
        '''
        :param frame:
        :param extra: number of frames in the first video
        :return:
        '''
        realframe = frame - extra
        time = (realframe / fps) / 60
        sec, min = math.modf(time)
        sec = sec * 60
        print('%s mins, %s secs' % (min, sec))

    def find_phase_starts(self, lst):
        phase_starts = [0]  # Start value of the first phase is always 0
        total = 0

        for value in lst:
            total += value
            phase_starts.append(total)
        phase_starts.pop()

        return phase_starts

    def find_blocks(self, array, gap_threshold, block_min_size):
        # mostly used in loco code
        blocks = []
        start = None
        for i in range(len(array) - 1):
            if start is None:
                start = array[i]
            if array[i + 1] - array[i] > gap_threshold:
                blocks.append((start, array[i]))
                start = None
        if start is not None:
            blocks.append((start, array[-1]))

        blocks = np.array(blocks)

        block_under_thresh = []
        for i in range(len(blocks)):
            b = blocks[i, 1] - blocks[i, 0] < block_min_size
            block_under_thresh.append(b)

        if np.any(blocks):
            blocks = blocks[~np.array(block_under_thresh)]

        return blocks

    def find_outliers(self, xdf, thresh=50, mask=False):
        neg_diff = xdf.diff()
        pos_diff = xdf.shift(-1).diff()
        outlier_mask = np.logical_and(abs(neg_diff) > thresh, abs(pos_diff) > thresh)
        outlier_idx = outlier_mask.index.get_level_values(level='FrameIdx')[outlier_mask==True]
        if mask:
            return outlier_mask
        else:
            return outlier_idx

    def Rotate2D(self, pts, cnt, ang):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return np.dot(pts-cnt,np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]))+cnt

    def picking_left_or_right(self, limb, comparison):
        """
        Function to return limb side given the side of the limb of interest and whether interested in contra- or ipsi-
        lateral limb
        :param limb: name of limb, eg 'L' or 'ForepawToeL'. Important thing is the format ends with the limb side L or R
        :param comparison: contr or ipsi
        :return: 'l' or 'r'
        """
        options = {
            'L': {
                'contr': 'R',
                'ipsi': 'L'
            },
            'R': {
                'contr': 'L',
                'ipsi': 'R'
            }
        }
        limb_lr = limb[-1]

        return options[limb_lr][comparison]

    def plot_polygon_with_numberered_pts(self, shape):
        x_side_coor, y_side_coor = zip(*shape)
        fig, ax = plt.subplots()
        polygon = Polygon(shape, closed=True, fill=None, edgecolor='b')
        ax.add_patch(polygon)
        for i, (x, y) in enumerate(shape):
            ax.text(x, y, str(i), color='red')

        ax.set_xlim(min(x_side_coor) - 1, max(x_side_coor) + 1)
        ax.set_ylim(min(y_side_coor) - 1, max(y_side_coor) + 1)
        fig.show()







