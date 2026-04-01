"""Compute per-run behavioural measures such as wait time, paw preference, and transition accuracy."""
import numpy as np
import re
import pandas as pd

from helpers.config import *

class CalculateMeasuresByRun():
    def __init__(self, XYZw, mouseID, r, stepping_limb, conditions):
        self.XYZw, self.mouseID, self.r, self.stepping_limb, self.conditions = XYZw, mouseID, r, stepping_limb, conditions

        # calculate sumarised dataframe
        self.data_chunk = self.XYZw[mouseID].loc(axis=0)[r]

    def wait_time(self):
        indexes = self.data_chunk.index.get_level_values('RunStage').unique()
        if 'TrialStart' in indexes:
            trial_start_idx = self.data_chunk.loc(axis=0)['TrialStart'].index[0]
            run_start_idx = self.data_chunk.loc(axis=0)['RunStart'].index[0]
            #run_start_idx = self.data_chunk.loc(axis=0)['TrialStart'].index[-1]
            duration_idx = run_start_idx - trial_start_idx
            return duration_idx/fps
        else:
            return 0

    def num_rbs(self, gap_thresh=30):
        if np.any(self.data_chunk.index.get_level_values(level='RunStage') == 'RunBack'):
            rb_chunk = self.data_chunk[self.data_chunk.index.get_level_values(level='RunStage') == 'RunBack']
            nose_tail = rb_chunk.loc(axis=1)['Nose', 'x'] - rb_chunk.loc(axis=1)['Tail1', 'x']
            nose_tail_bkwd = nose_tail[nose_tail < 0]
            num = sum(np.diff(nose_tail_bkwd.index.get_level_values(level='FrameIdx')) > gap_thresh)
            return num + 1
        else:
            return 0

    def start_paw_pref(self):
        start_idx = self.data_chunk.loc(axis=0)['RunStart'].index[0]
        initiating_paw = self.data_chunk.loc(axis=0)['RunStart',start_idx].loc['initiating_limb'].values[0]
        return micestuff['LR'][initiating_paw]

    def trans_paw_pref(self):
        if self.stepping_limb == 'ForepawL' or self.stepping_limb == 'ForepawL_slid':
            return micestuff['LR']['ForepawL']
        elif self.stepping_limb == 'ForepawR' or self.stepping_limb == 'ForepawR_slid':
            return micestuff['LR']['ForepawR']

    def start_to_trans_paw_matching(self):
        if self.start_paw_pref() == self.trans_paw_pref():
            return True
        else:
            return False

    def post_transition_hit_position(self):
        """
        Calculate the distance in x between the transition point and the paw hit position
        :return:
        """
        transition_idx = self.data_chunk.loc(axis=0)['Transition'].index[0]

        limb_name_clean = re.sub(r'_.*$', '', self.stepping_limb)
        pattern = r'^(.*?)([RL])$'
        replacement = r'\1' + 'Toe' + r'\2'
        limb_name_modified = re.sub(pattern, replacement, limb_name_clean)

        transition_x = self.data_chunk.loc(axis=0)['Transition',transition_idx].loc[limb_name_modified,'x']

        distance_from_transition = transition_x - expstuff['setup']['transition_mm']
        return distance_from_transition

    def pre_transition_hit_position(self):
        """
        Calculate the distance in x between the transition point and the paw hit position
        :return:
        """
        limb_name_clean = re.sub(r'_.*$', '', self.stepping_limb)
        pattern = r'^(.*?)([RL])$'
        replacement = r'\1' + 'Toe' + r'\2'
        limb_name_modified = re.sub(pattern, replacement, limb_name_clean)

        # find swing start before transition for stepping limb
        swst = self.data_chunk.loc(axis=0)['RunStart'].loc(axis=1)[self.stepping_limb,'SwSt_discrete']
        swing_start_idx =  swst[swst == locostuff['swst_vals_2025']['sw']].index[-1]

        swing_start_x = self.data_chunk.loc(axis=0)['RunStart',swing_start_idx].loc[limb_name_modified,'x']
        distance_from_transition = expstuff['setup']['transition_mm'] - swing_start_x
        return distance_from_transition

    def length_transitioning_swing(self):
        pre = self.pre_transition_hit_position()
        post = self.post_transition_hit_position()
        length = pre + post
        return length

    # todo detect 2x hindpaw leaps at transition
    # todo detect sitting/standing

    def run(self):
        column_names = ['wait_time','num_rbs','start_paw_pref','trans_paw_pref','start_to_trans_paw_matching','post_transition_hit_position','pre_transition_hit_position','length_transitioning_swing']
        index_names = pd.MultiIndex.from_tuples([(self.mouseID,self.r)], names=['MouseID','Run'])
        df = pd.DataFrame(index=index_names, columns=column_names)
        data = np.array([self.wait_time(),self.num_rbs(),self.start_paw_pref(),self.trans_paw_pref(),self.start_to_trans_paw_matching(),self.post_transition_hit_position(),self.pre_transition_hit_position(),self.length_transitioning_swing()])
        df.loc[self.mouseID, self.r] = data
        return df