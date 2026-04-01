"""Compute per-stride kinematic measures including timing, velocity, displacement, and joint angles."""
import warnings
import re
import pandas as pd
import itertools
from scipy.signal import correlate
from scipy.signal import savgol_filter

from helpers.config import *
from helpers import utils

class CalculateMeasuresByStride():
    def __init__(self, XYZw, mouseID, r, stride_start, stride_end, stepping_limb, conditions):
        self.XYZw, self.mouseID, self.r, self.stride_start, self.stride_end, self.stepping_limb, self.conditions = \
            XYZw, mouseID, r, stride_start, stride_end, stepping_limb, conditions

        # calculate sumarised dataframe
        self.data_chunk = self.XYZw[mouseID].loc(axis=0)[r, ['RunStart', 'Transition', 'RunEnd']].droplevel(['Run', 'RunStage']).loc(axis=0)[np.arange(self.stride_start ,self.stride_end +1)]

    ####################################################################################################################
    ### General utility functions
    ####################################################################################################################

    def calculate_belt_speed(self): # mm/s
        speed_condition = self.conditions['speed']
        run_phases = expstuff['condition_exp_runs_basic'][self.conditions['exp']][self.conditions['repeat_extend']]
        for phase_name, run_array in run_phases.items():
            if self.r in run_array:
                run_phase = phase_name
        if 'Baseline' in run_phase or 'Washout' in run_phase:
            belt1_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[0]] * 10 # convert to mm
            belt2_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[0]] * 10 # convert to mm
        elif 'APA' in run_phase:
            belt1_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[0]] * 10 # convert to mm
            belt2_speed = expstuff['speeds'][re.findall(r'[A-Z][^A-Z]*', speed_condition)[1]] * 10 # convert to mm
        else:
            raise ValueError(f'Run phase {run_phase} not recognised')

        transition_idx = self.XYZw[self.mouseID].loc(axis=0)[self.r ,'Transition'].index[0]
        belt_speed = belt1_speed if self.stride_start < transition_idx else belt2_speed

        return belt_speed

    def calculate_belt_x_displacement(self):
        belt_speed = self.calculate_belt_speed()
        time_s = (self.stride_end - self.stride_start ) /fps
        distance_mm = belt_speed * time_s # mm/s / s
        return distance_mm

    def get_buffer_chunk(self, buffer_size):
        """
        Get stride data with x% of stride length in buffer frames either side
        :param buffer_size: percentage as decimal of stride length that want in frames either side of start and end of stride
        :return: the new data chunk
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)
        stride_length = self.stride_end - self.stride_start
        start = self.stride_start - round(stride_length * buffer_size)
        end = self.stride_end + round(stride_length * buffer_size)
        buffer_chunk = self.XYZw[self.mouseID].loc(axis=0)[
            self.r, ['RunStart', 'Transition', 'RunEnd'], np.arange(start, end)]
        return buffer_chunk

    def convert_notoe_to_toe(self, limb_name):
        limb_name_clean = re.sub(r'_.*$', '', limb_name)
        pattern = r'^(.*?)([RL])$'
        replacement = r'\1' + 'Toe' + r'\2'
        limb_name_modified = re.sub(pattern, replacement, limb_name_clean)
        return limb_name_modified

    def convert_toe_to_notoe(self, limb_name):
        # Step 1: Remove any trailing suffix starting with an underscore
        limb_name_clean = re.sub(r'_.*$', '', limb_name)

        # Step 2: Remove 'Toe' before the directional suffix ('R' or 'L')
        pattern = r'^(.*?)(Toe)([RL])$'
        replacement = r'\1\3'  # Replace with the first and third captured groups
        limb_name_modified = re.sub(pattern, replacement, limb_name_clean)

        return limb_name_modified

    ####################################################################################################################
    ### Single value only calculations
    ####################################################################################################################

    ########### DURATIONS ###########:

    def stride_duration(self):
        """
        :return: Duration in seconds
        """
        stride_frames = self.data_chunk.index[-1] - self.data_chunk.index[0]
        return (stride_frames / fps) # * 1000

    def stance_duration(self):
        """
        :return: Duration in seconds
        """
        stance_mask = self.data_chunk.loc(axis=1)[self.stepping_limb ,'SwSt'] == locostuff['swst_vals_2025']['st']
        stance = self.data_chunk.loc(axis=1)[self.stepping_limb][stance_mask]
        stance_frames = stance.index[-1] - stance.index[0]
        return (stance_frames / fps) # * 1000

    def swing_duration(self):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        swing = self.data_chunk.loc(axis=1)[self.stepping_limb][swing_mask]
        swing_frames = swing.index[-1] - swing.index[0]
        return (swing_frames / fps)  # * 1000

    def cadence(self):
        stride_duration = self.stride_duration()
        return 1/ stride_duration

    def duty_factor(self):  # %
        stance_duration = self.stance_duration()
        stride_duration = self.stride_duration()
        result = (stance_duration / stride_duration) * 100
        return result

    def brake_prop_duration(self, type):
        """
        Returns the duration (in seconds) of the braking phase during stance. Here, braking is defined
        as the period (in stance) from the initial touchdown (first frame where the stepping limb is in stance)
        until the forward (x-axis) velocity of the stepping limb first becomes nonnegative.
        !!!!! NOT TRUE !!!!! it's the time of maximal contact in z
        """
        # Select stance-phase frames for the stepping limb
        stance_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['st']
        chunk = self.data_chunk[stance_mask]

        # Use the frame number from the multi-index.
        # (Assuming the index level 'FrameIdx' holds frame numbers.)
        dt = 1 / fps  # Convert frames to seconds.
        initial_frame = chunk.index.get_level_values('FrameIdx')[0]
        initial_time = initial_frame * dt

        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        # Define the landmark labels for the stepping limb.
        toe_label = f'ForepawToe{lr}'
        knuckle_label = f'ForepawKnuckle{lr}'
        ankle_label = f'ForepawAnkle{lr}'

        # Extract the z-coordinate series for each landmark from the chunk.
        toe_z = chunk.loc(axis=1)[toe_label, 'z']
        knuckle_z = chunk.loc(axis=1)[knuckle_label, 'z']
        ankle_z = chunk.loc(axis=1)[ankle_label, 'z']

        # Compute an overall contact metric as the average z-value (the lower, the better contact).
        # (If the belt is at z==0, then lower z means closer to the belt.)
        contact_metric = (toe_z + knuckle_z + ankle_z) / 3.0

        # Identify the frame (within the chunk) where contact is maximal (i.e. minimal average z).
        max_contact_idx = contact_metric.idxmin()  # This returns a multi-index for that frame.
        # Extract the frame number from the multi-index. (Assumes 'FrameIdx' is one level.)
        max_time = max_contact_idx * dt

        # Next, determine swing onset.
        # We search in the full data (or a larger slice) for the first frame after max_contact
        # where the stepping limb’s state becomes swing.
        swing_frame_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']
        swing_frame = self.data_chunk[swing_frame_mask].index[0]
        swing_time = swing_frame * dt

        if type == 'brake':
            brake_time = max_time - initial_time
            return brake_time
        elif type == 'propulsion':
            propulsion_time = swing_time - max_time
            return propulsion_time

    # todo could try include tau propulsion also but it seems my x velocities are not precise enough to get a good estimate

    ########### SPEEDS ###########:

    def walking_speed(self, bodypart, speed_correct):
        """
        :param bodypart: Either Tail1 or Back6
        :return: Speed in mm/s
        """
        x_displacement = self.data_chunk.loc(axis=1)[bodypart, 'x'].iloc[-1] - \
                         self.data_chunk.loc(axis=1)[bodypart, 'x'].iloc[0]  # mm
        walking_speed_mm_s = x_displacement / self.stride_duration()
        if not speed_correct:
            return walking_speed_mm_s
        else:
            return walking_speed_mm_s - self.calculate_belt_speed()

    def swing_velocity(self, speed_correct):
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        swing = self.data_chunk.loc(axis=1)[limb_name][swing_mask]
        swing_length = swing.loc(axis=1)['x'].iloc[-1] - swing.loc(axis=1)['x'].iloc[0]
        swing_frames = swing.index[-1] - swing.index[0]
        swing_duration = (swing_frames / fps)  # * 1000
        swing_vel = swing_length / swing_duration
        if not speed_correct:
            return swing_vel
        else:
            return swing_vel - self.calculate_belt_speed()

    def instantaneous_swing_velocity(self, speed_correct, xyz, smooth=False):
        """
        Derivative of swing trajectory
        :return: dataframe of velocities for x, y and z
        """
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        swing_trajectory = self.traj(limb_name, coord=xyz, step_phase='0', full_stride=False,
                                     speed_correct=False, aligned=False,
                                     buffer_size=0)  ####todo check speed correct should be false!
        time_interval = self.swing_duration()
        d_xyz = swing_trajectory.diff()
        v_xyz = d_xyz / time_interval
        if smooth:
            # Optionally, smooth the velocities using Savitzky-Golay filter
            v_xyz = savgol_filter(v_xyz, window_length=3, polyorder=1)
        if not speed_correct:
            return v_xyz
        else:
            if xyz == 'x':
                return v_xyz - self.calculate_belt_speed()
            else:
                return v_xyz

    ########### DISTANCES ###########:

    def stride_length(self, speed_correct):
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        length = self.data_chunk.loc(axis=1)[limb_name, 'x'].iloc[-1] - \
                 self.data_chunk.loc(axis=1)[limb_name, 'x'].iloc[0]
        if not speed_correct:
            return length
        else:
            return length - self.calculate_belt_x_displacement()

    def net_displacement_rel(self, coord, bodypart, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        The difference in bodypart coordinates in x, y, z relative to tailbase from start to end of step phase.
        """
        if bodypart == 'FrontIpsi':
            bodypart = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        elif bodypart == 'FrontContra':
            bodypart = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        elif bodypart == 'HindIpsi':
            bodypart = 'HindpawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        elif bodypart == 'HindContra':
            bodypart = 'HindpawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        else:
            bodypart = bodypart

        # relative to Tail1 frame by frame
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            val = buffer_chunk.loc(axis=1)[bodypart, coord]
            tail = buffer_chunk.loc(axis=1)['Tail1', coord]
            rel_val = val - tail
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            val = self.data_chunk.loc(axis=1)[bodypart, coord][stsw_mask]
            tail = self.data_chunk.loc(axis=1)['Tail1', coord][stsw_mask]
            rel_val = val - tail
        #x = x - self.calculate_belt_x_displacement() if speed_correct else x # dont do this anymore as relative to tail anyway
        if all_vals:
            return rel_val.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return rel_val.iloc[-1] - rel_val.iloc[0]

    def distance_to_transition(self, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        Proximity of tailbase (to which other displacement measures are referenced to) at beginning of step phase.
        A positive value indicates that the tailbase is ahead of the transition point.
        """
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            proximity = buffer_chunk.loc(axis=1)['Tail1', 'x'] - 470
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            proximity = self.data_chunk.loc(axis=1)['Tail1', 'x'][stsw_mask] - 470
        if all_vals:
            return proximity.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return proximity.iloc[0]

    def distance_from_midline(self, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        Proximity of tailbase (to which other displacement measures are referenced to) at beginning of step phase.
        A positive value indicates that the tailbase is to the right of the midline.
        """
        y_midline = structural_stuff['belt_width'] / 2
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            proximity = buffer_chunk.loc(axis=1)['Tail1', 'y'] - y_midline
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            proximity = self.data_chunk.loc(axis=1)['Tail1', 'y'][stsw_mask] - y_midline
        if all_vals:
            return proximity.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return proximity.iloc[0]

    def distance_from_belt_surface(self, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        Proximity of tailbase to belt surface (z==0) at beginning of step phase.
        """
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            proximity = buffer_chunk.loc(axis=1)['Tail1', 'z']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            proximity = self.data_chunk.loc(axis=1)['Tail1', 'z'][stsw_mask]
        if all_vals:
            return proximity.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return proximity.iloc[0]

    def excursion(self, bodypart, coord, buffer_size=0.25):
        """
        the overall range of movement during a stride—that is, the difference between the highest and lowest x/y/z-values reached.
        # todo NB this is NOT relative to tailbase (currently)
        """
        if bodypart == 'FrontIpsi':
            bodypart = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        elif bodypart == 'FrontContra':
            bodypart = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        elif bodypart == 'HindIpsi':
            bodypart = 'HindpawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        elif bodypart == 'HindContra':
            bodypart = 'HindpawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        else:
            bodypart = bodypart

        buffer_chunk = self.get_buffer_chunk(buffer_size)
        val = buffer_chunk.loc(axis=1)[bodypart, coord]
        return val.max() - val.min()


    def traj(self, bodypart, coord, step_phase, full_stride, speed_correct, aligned, buffer_size=0.25):
        if bodypart == 'FrontIpsi':
            bodypart = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        elif bodypart == 'FrontContra':
            bodypart = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        elif bodypart == 'HindIpsi':
            bodypart = 'HindpawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
        elif bodypart == 'HindContra':
            bodypart = 'HindpawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        else:
            bodypart = bodypart

        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            xyz = buffer_chunk.loc(axis=1)[bodypart, ['x', 'y', 'z']].droplevel('bodyparts', axis=1).droplevel(
                ['Run', 'RunStage'], axis=0)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            xyz = self.data_chunk.loc(axis=1)[bodypart, ['x', 'y', 'z']][stsw_mask].droplevel('bodyparts', axis=1)
        if speed_correct:
            xyz['x'] = xyz['x'] - self.calculate_belt_x_displacement()
        if aligned:
            xyz = xyz - xyz.loc(axis=0)[
                self.stride_start]  ### todo cant align traj to first position when that is a nan
        return xyz[coord]

    ########### BODY-RELATVE DISTANCES/ coordination ###########:

    def coo_xyz(self, xyz):
        """
        Centre of oscillation in x, y, OR z of paw with respect to Back6
        :param xyz:
        :return:
        """
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        mid_back = swing.loc(axis=1)['Back6', xyz]  # .loc(axis=0)[mid_t]
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        limb = swing.loc(axis=1)[limb_name, xyz]

        return limb[mid_t] - mid_back[mid_t]

    def coo_euclidean(self):
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        swing_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['sw']
        swing = self.data_chunk[swing_mask]
        mid_t = np.median(swing.index).astype(int)
        bodypart1 = swing.loc(axis=1)['Back6', ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        bodypart2 = swing.loc(axis=1)[limb_name, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        bodypart1 = bodypart1.loc(axis=0)[mid_t]
        bodypart2 = bodypart2.loc(axis=0)[mid_t]
        distance = np.sqrt((bodypart2['x'] - bodypart1['x']) ** 2 + (bodypart2['y'] - bodypart1['y']) ** 2 + (
                    bodypart2['z'] - bodypart1['z']) ** 2)
        return distance

    def bos_stancestart(self, ref_or_contr, y_or_euc):
        """
        Base of support - Y distance between front paws at start of *stepping limb* or *contralateral limb* stance
        :param all_vals: If true, returns all values from stride
        @:param ref_or_contr: which limb stance start use as timepoint for analysis
        :return (float): base of support
        """
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
        stance_limb = self.stepping_limb if ref_or_contr == 'ref' else 'Forepaw%s' % lr
        st_mask = self.data_chunk.loc(axis=1)[stance_limb, 'SwSt'] == locostuff['swst_vals_2025']['st']
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        steppinglimb = self.data_chunk.loc(axis=1)[limb_name][st_mask]
        contrlimb = self.data_chunk.loc(axis=1)['ForepawToe%s' % lr][st_mask]
        if y_or_euc == 'y':
            bos = abs(steppinglimb['y'] - contrlimb['y'])
        else:
            bos = np.sqrt((contrlimb['x'] - steppinglimb['x']) ** 2 + (contrlimb['y'] - steppinglimb['y']) ** 2 + (
                        contrlimb['z'] - steppinglimb['z']) ** 2)
        return bos.values[0]

    def ptp_amplitude_stride(self, bodypart):
        coords = self.data_chunk.loc(axis=1)[bodypart, 'z']
        peak = coords.max()
        trough = coords.min()
        return peak - trough

    def body_length(self, bodyparts, step_phase, all_vals, full_stride,
                      buffer_size=0.25):  ### eg body length, midback to forepaw
        bodypart1, bodypart2 = bodyparts
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x1 = buffer_chunk.loc(axis=1)[bodypart1, 'x']
            x2 = buffer_chunk.loc(axis=1)[bodypart2, 'x']
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            x1 = self.data_chunk.loc(axis=1)[bodypart1, 'x'][stsw_mask]
            x2 = self.data_chunk.loc(axis=1)[bodypart2, 'x'][stsw_mask]
        length = abs(x1 - x2)
        if all_vals:
            return length.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return length.mean()

    def back_height(self, back_label, step_phase, all_vals, full_stride, buffer_size=0.25):
        # back_labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
        #                'Back11', 'Back12']
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            back_heights = buffer_chunk.loc(axis=1)[back_label, 'z'].droplevel(['Run', 'RunStage'],
                                                                                axis=0)  # .droplevel(level='coords', axis=1).iloc[:, ::-1])
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            back_heights = self.data_chunk.loc(axis=1)[back_label, 'z'][
                stsw_mask]  # .droplevel(level='coords', axis=1).iloc[:, ::-1])
        if all_vals:
            # return back_heights.droplevel(['Run','RunStage'],axis=0)
            return back_heights
        else:
            return back_heights.mean(axis=0)

    def tail_height(self, tail_label, step_phase, all_vals, full_stride, buffer_size=0.25):
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            tail_heights = buffer_chunk.loc(axis=1)[tail_label, 'z'].droplevel(['Run', 'RunStage'],
                                                                                axis=0)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            tail_heights = self.data_chunk.loc(axis=1)[tail_label, 'z'][
                stsw_mask]  # .droplevel(level='coords', axis=1).iloc[:, ::-1])
        if all_vals:
            # return back_heights.droplevel(['Run','RunStage'],axis=0)
            return tail_heights
        else:
            return tail_heights.mean(axis=0)

        # todo body y dislacment
        # todo body pitch, roll, yaw

    ########### BODY-RELATIVE TIMINGS/PHASES/coordination ###########:

    def double_support(self, support_type):
        """
        Calculates the percentage of the stride cycle between the touch-down of the reference forepaw and the lift-off
        of a second limb, based on the provided support type.

        support_type:
            'frontonly'   => Contralateral forepaw swing (may include negative double support)
            'homolateral' => Ipsilateral hindpaw swing
            'diagonal'    => Contralateral hindpaw swing

        Returns:
            A percentage value of the stride cycle (positive if swing occurs after reference touchdown, negative if before).
        """
        # Choose which limb and paw to use based on support_type
        if support_type == 'frontonly':
            limb_direction = 'contr'
            paw = 'Forepaw'
        elif support_type == 'homolateral':
            limb_direction = 'ipsi'
            paw = 'Hindpaw'
        elif support_type == 'diagonal':
            limb_direction = 'contr'
            paw = 'Hindpaw'
        else:
            raise ValueError(f"Invalid support_type provided: {support_type}")

        # Determine the correct left/right indicator for the chosen limb
        lr = utils.Utils().picking_left_or_right(self.stepping_limb, limb_direction)

        # Compute stride length and prepare the frame range for negative swing detection
        stride_length = self.stride_end - self.stride_start
        # Here we consider frames starting up to 25% of the stride before the stride start
        neg_frame_range = np.arange(self.stride_start - round(stride_length / 4), self.stride_start)

        # Positive swing mask: within the stride cycle (data_chunk should be the current stride’s chunk)
        pos_mask = self.data_chunk.loc(axis=1)[f'{paw}{lr}', 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']

        # Negative swing mask: before the stride cycle from the complete XYZw data
        neg_mask = self.XYZw[self.mouseID].loc(axis=0)[self.r, ['RunStart', 'Transition', 'RunEnd'], neg_frame_range] \
                       .loc(axis=1)[f'{paw}{lr}', 'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']

        # Calculate result based on the first occurrence in either mask
        if any(pos_mask):
            # For the positive mask, get the first matching frame from the data_chunk's index
            swing_frame = self.data_chunk.index.get_level_values('FrameIdx')[pos_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            #stride_duration = self.stride_duration()
            result = ((swing_frame - ref_stance_frame) / stride_length) * 100  # old : ((swing_frame - ref_stance_frame) / stride_duration) * 100
        elif any(neg_mask):
            # For the negative mask, get the first matching frame from the XYZw data (within the negative frame range)
            swing_frame = self.XYZw[self.mouseID].loc(axis=0)[
                self.r, ['RunStart', 'Transition', 'RunEnd'], neg_frame_range] \
                .index.get_level_values('FrameIdx')[neg_mask][0]
            ref_stance_frame = self.data_chunk.index[0]
            #stride_duration = self.stride_duration()
            result = ((swing_frame - ref_stance_frame) / stride_length) * 100  # old : ((swing_frame - ref_stance_frame) / stride_duration) * 100
        else:
            result = 0

        return result

    def triple_support(self, mode):
        """
        Calculates the percentage of the stride cycle during which triple support occurs.

        Parameters:
            mode: str, either 'any' or 'front+hind'
                - 'any': returns the percentage of frames where any three (or four) limbs are in stance.
                - 'front+hind': returns the percentage of frames where both forepaws are in stance
                                and at least one hindpaw is in stance.

        Returns:
            Percentage (0-100) of the stride cycle with triple support.
        """
        # Define the stance value and the list of limbs
        stance_val = locostuff['swst_vals_2025']['st']
        limbs = ['ForepawL', 'ForepawR', 'HindpawL', 'HindpawR']

        # Create a boolean DataFrame: each cell True if that limb is in stance at that frame.
        stance_df = pd.DataFrame({
            limb: (self.data_chunk.loc(axis=1)[limb, 'SwSt'] == stance_val)
            for limb in limbs
        }, index=self.data_chunk.index)

        if mode == 'any':
            # Count frames where at least three of the four limbs are in stance.
            triple_frames = stance_df.sum(axis=1) >= 3
        elif mode == 'front_hind':
            # Require that both forepaws are in stance and at least one hindpaw is in stance.
            front_support = stance_df['ForepawL'] & stance_df['ForepawR']
            hind_support = stance_df['HindpawL'] | stance_df['HindpawR']
            triple_frames = front_support & hind_support
        else:
            raise ValueError("Invalid mode for triple_support. Use 'any' or 'front+hind'.")

        # Calculate percentage: (# of frames meeting condition) / (total frames) * 100
        percent_triple = triple_frames.sum() / len(triple_frames) * 100
        return percent_triple

    def quadruple_support(self):
        """
        Calculates the percentage of the stride cycle during which all four paws are in support.

        Returns:
            Percentage (0-100) of the stride cycle with quadruple (all four) support.
        """
        # Define the stance value and the list of limbs
        stance_val = locostuff['swst_vals_2025']['st']
        limbs = ['ForepawL', 'ForepawR', 'HindpawL', 'HindpawR']

        # Create a boolean DataFrame for each limb's stance state.
        stance_df = pd.DataFrame({
            limb: (self.data_chunk.loc(axis=1)[limb, 'SwSt'] == stance_val)
            for limb in limbs
        }, index=self.data_chunk.index)

        # Identify frames where all four limbs are in stance.
        quadruple_frames = (stance_df.sum(axis=1) == 4)

        # Calculate percentage of stride cycle with quadruple support.
        percent_quadruple = quadruple_frames.sum() / len(quadruple_frames) * 100
        return percent_quadruple

    def stance_phase(self, stance_limb):
        """relative timing of limb touchdowns to stride cycle of reference paw (FR). Calculated as: stance time - stance
        timereference paw/stride duration.
        N.B. only measuring length of stance time *within* the stepping limbs stride cycle
        """
        if stance_limb == 'contra_front':
            limb_direction = 'contr'
            paw = 'Forepaw'
        elif stance_limb == 'contra_hind':
            limb_direction = 'contr'
            paw = 'Hindpaw'
        elif stance_limb == 'ipsi_hind':
            limb_direction = 'ipsi'
            paw = 'Hindpaw'
        else:
            raise ValueError(f"Invalid stance_limb provided: {stance_limb}")

        stride_length = self.stride_end - self.stride_start

        lr = utils.Utils().picking_left_or_right(self.stepping_limb, limb_direction)
        stance_limb = f'{paw}{lr}'
        stance_limb_mask = self.data_chunk.loc(axis=1)[stance_limb, 'SwSt'] == locostuff['swst_vals_2025']['st']
        stepping_limb_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == locostuff['swst_vals_2025']['st']
        stance_limb_length = len(self.data_chunk.loc(axis=1)[stance_limb, 'SwSt'][stance_limb_mask])
        stepping_limb_length = len(self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'][stepping_limb_mask])
        stance_phase = (stance_limb_length - stepping_limb_length) / stride_length

        return stance_phase

    def nose_tail_phase(self, bodypart, frontback, coord):
        """
        This function is essentially measuring the timing offset—often called the "phase"—between the movement of a
        body part (like the nose or tail) and how the two front or two hind paws move during a stride.

        Calculate the phase (delay) at which the stridewise trajectory of the specified body part
        (e.g. 'nose' or 'tail') maximally correlates with the reference trajectory defined as the difference
        between the forward positions of the right and left paws. This is computed separately for front
        limbs (ForepawR - ForepawL) and hind limbs (HindpawR - HindpawL).

        Parameters:
            bodypart (str): The body part to analyze (e.g. 'nose' or 'tail').
            coordinate (str): The coordinate to use (e.g. 'x').

        Returns:
            dict: A dictionary with keys:
                  'phase_front' - the phase delay (in normalized time units) for the front limb reference,
                  'phase_hind'  - the phase delay for the hind limb reference.
        """
        # Number of points to which trajectories are normalized (adjust as desired)
        num_points = 100
        dt = 1 / (num_points - 1)

        def normalize_trajectory(traj, num_points=num_points):
            """
            Normalize a 1D trajectory to a fixed number of points using linear interpolation.
            Assumes the original trajectory spans a normalized time from 0 to 1.
            """
            traj = np.asarray(traj)
            x_orig = np.linspace(0, 1, len(traj))
            x_new = np.linspace(0, 1, num_points)
            return np.interp(x_new, x_orig, traj)

        def compute_phase(traj1, traj2, dt):
            """
            Compute the delay (phase) that maximizes the cross-correlation between two 1D signals.
            """
            corr = correlate(traj1, traj2, mode='full')
            lags = np.arange(-len(traj1) + 1, len(traj1))
            best_lag = lags[np.argmax(corr)]
            return best_lag * dt

        # Get the trajectory of the specified body part from the current stride.
        # (self.data_chunk is assumed to contain frames from stride_start to stride_end.)
        body_traj = self.data_chunk.loc(axis=1)[bodypart, coord].values
        body_traj_norm = normalize_trajectory(body_traj, num_points)

        if frontback == 'front':
            # Compute the front paw difference trajectory: (ForepawR - ForepawL)
            right_fore = self.data_chunk.loc(axis=1)['ForepawToeR', coord].values
            left_fore = self.data_chunk.loc(axis=1)['ForepawToeL', coord].values
            front_diff = right_fore - left_fore
            front_diff_norm = normalize_trajectory(front_diff, num_points)

            # compute the phase delays via cross-correlation
            phase_front = compute_phase(body_traj_norm, front_diff_norm, dt)
            return phase_front

        elif frontback == 'hind':
            # Compute the hind paw difference trajectory: (HindpawR - HindpawL)
            right_hind = self.data_chunk.loc(axis=1)['HindpawToeR', coord].values
            left_hind = self.data_chunk.loc(axis=1)['HindpawToeL', coord].values
            hind_diff = right_hind - left_hind
            hind_diff_norm = normalize_trajectory(hind_diff, num_points)

            # Compute the phase delays via cross-correlation.
            phase_hind = compute_phase(body_traj_norm, hind_diff_norm, dt)
            return phase_hind


    ########### BODY-RELATIVE POSITIONING ###########:
    def back_skew(self, step_phase, all_vals, full_stride,
                  buffer_size=0.25):  ##### CHECK HOW TO DEAL WITH MISSING BACK VALUES - HAVE A MULT ROW FOR EVERY FRAME BASED ON HOW MANY TRUE VALUES I HAVE
        back_labels = ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10',
                       'Back11', 'Back12']
        mult = np.arange(1, 13)
        true_back_height = []
        for b in reversed(back_labels):
            true_back_height_label = self.back_height(b, step_phase, all_vals=True, full_stride=full_stride,
                                                      buffer_size=buffer_size)
            true_back_height.append(true_back_height_label)
        true_back_height = pd.concat(true_back_height, axis=1)
        COM = (true_back_height * mult).sum(axis=1) / true_back_height.sum(axis=1)  # calculate centre of mass
        skew = np.median(mult) - COM
        if all_vals:
            return skew
        else:
            return skew.mean()

    def limb_rel_to_body(self, time, step_phase, all_vals, full_stride,
                         buffer_size=0.25):  # back1 is 1, back 12 is 0, further forward than back 1 is 1+
        """
        The relative position of the stepping limb to the body, normalized to the distance between Back1 and Back12.
        A value of 0 indicates that the limb is at the same position as Back12, while a value of 1 indicates that
        the limb is at the same position as Back1. Values greater than 1 indicate that the limb is further forward
        than Back1.
        """
        limb_name = self.convert_notoe_to_toe(self.stepping_limb)
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            x = buffer_chunk.loc(axis=1)[['Back1', 'Back12', limb_name], 'x'].droplevel(['Run', 'RunStage'],
                                                                                                 axis=0)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            x = self.data_chunk.loc(axis=1)[['Back1', 'Back12', limb_name], 'x'][stsw_mask]
        x_zeroed = x - x['Back12']
        x_norm_to_neck = x_zeroed / x_zeroed['Back1']
        position = x_norm_to_neck[limb_name]
        position = pd.Series(position.values.flatten(), index=position.index)
        if all_vals:
            return position
        else:
            if time == 'start':
                return position.iloc[0]
            elif time == 'end':
                return position.iloc[-1]

    def signed_angle(self, reference_vector, plane_normal, bodyparts, step_phase, all_vals, full_stride, summary_stats,
                     buffer_size=0.25):
        """
        Calculates the signed angle between the vector from bodypart1 to bodypart2 and a reference vector
        when viewed from a given plane. **Positive angles are clockwise and negative anticlockwise from the reference.**
        If all_vals is True, returns a pd.Series of angles (in degrees) indexed by FrameIdx.
        Otherwise, returns a dictionary with two single values:
            - 'average': the average signed angle over the selected frames.
            - 'peak_amplitude': the maximum absolute angle (peak excursion).

        Parameters:
            reference_vector (np.array): The reference vector within the plane.
            plane_normal (np.array): The normal vector of the plane.
            bodyparts (tuple): A tuple containing two strings: (bodypart1, bodypart2).
            step_phase (str): The step phase to use when full_stride is False.
            all_vals (bool): If True, returns all frame values; if False, returns summary statistics.
            full_stride (bool): If True, uses the full stride (with a buffer); if False, uses only frames where
                                self.stepping_limb is in the specified step_phase.
            buffer_size (float): Proportion of stride to include as buffer.

        Returns:
            Either a pd.Series (if all_vals True) or a dict with keys 'average' and 'peak_amplitude'.
        """
        if bodyparts == ['FrontAnkleIpsi', 'FrontToeIpsi']:
            bodypart1 = 'ForepawAnkle%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
            bodypart2 = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
            bodyparts = [bodypart1, bodypart2]
        elif bodyparts == ['FrontAnkleContra', 'FrontToeContra']:
            bodypart1 = 'ForepawAnkle%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
            bodypart2 = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
            bodyparts = [bodypart1, bodypart2]
        elif bodyparts == ['FrontKnuckleIpsi', 'FrontToeIpsi']:
            bodypart1 = 'ForepawKnuckle%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
            bodypart2 = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'ipsi')
            bodyparts = [bodypart1, bodypart2]
        elif bodyparts == ['FrontKnuckleContra', 'FrontToeContra']:
            bodypart1 = 'ForepawKnuckle%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
            bodypart2 = 'ForepawToe%s' % utils.Utils().picking_left_or_right(self.stepping_limb, 'contr')
            bodyparts = [bodypart1, bodypart2]
        else:
            bodyparts = bodyparts


        # Select the appropriate data chunk
        if full_stride:
            data_chunk = self.get_buffer_chunk(buffer_size)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            data_chunk = self.data_chunk[stsw_mask]

        # Extract coordinates for the two body parts and drop the bodyparts level
        bodypart1, bodypart2 = bodyparts
        coord_1 = data_chunk.loc(axis=1)[bodypart1, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        coord_2 = data_chunk.loc(axis=1)[bodypart2, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)

        # Convert to numpy arrays
        A = coord_1.to_numpy()
        B = coord_2.to_numpy()
        # Compute vectors from A to B
        vectors_AB = B - A

        # Project vectors_AB onto the plane by removing the component along plane_normal
        vectors_AB_projected = vectors_AB - np.outer(
            np.dot(vectors_AB, plane_normal) / np.linalg.norm(plane_normal) ** 2,
            plane_normal)

        # Project the reference_vector onto the plane
        reference_vector_projected = reference_vector - (np.dot(reference_vector, plane_normal) /
                                                         np.linalg.norm(plane_normal) ** 2) * plane_normal

        # Normalize the projected vectors
        vectors_norm = np.linalg.norm(vectors_AB_projected, axis=1)
        # Avoid division by zero
        vectors_AB_projected_normalized = vectors_AB_projected / vectors_norm[:, np.newaxis]
        ref_norm = np.linalg.norm(reference_vector_projected)
        reference_vector_projected_normalized = reference_vector_projected / ref_norm

        # Compute the angle (in radians) via the dot product
        dot_products = np.dot(vectors_AB_projected_normalized, reference_vector_projected_normalized)
        angles_rad = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Determine the sign of each angle using the cross product and plane_normal
        cross_products = np.cross(vectors_AB_projected_normalized, reference_vector_projected_normalized)
        angle_signs = np.sign(np.dot(cross_products, plane_normal))
        signed_angles_rad = angles_rad * angle_signs
        signed_angles_deg = np.degrees(signed_angles_rad).flatten()

        # Create a Series with the FrameIdx from the coordinate index
        angle_series = pd.Series(signed_angles_deg, index=coord_1.index.get_level_values('FrameIdx'))

        if all_vals:
            return angle_series
        else:
            # Calculate summary statistics
            if summary_stats == 'mean':
                return angle_series.mean()
            elif summary_stats == 'peak':
                return np.max(np.abs(angle_series))


    def angle_3d(self, bodypart1, bodypart2, reference_axis, step_phase, all_vals, full_stride, buffer_size=0.25):
        """
        Calculate the angle between two body parts relative to a reference axis.
        :param bodypart1 (str): First body part
        :param bodypart2 (str): Second body part
        :param reference_axis (array): Reference axis in the form of a 3D vector.
        :param step_phase: 0 or 1 for swing or stance , respectively
        :param all_vals: True or False for returning all values in stride or averaging, respectively
        :param full_stride: True or False for analysing all frames from the stride and not splitting into st or sw
        :param buffer_size: Proportion of stride in franes to add before and end as a buffer, 0 to 1
        :return (angle): Angle between the two body parts and the reference axis (in degrees).
        """
        if full_stride:
            buffer_chunk = self.get_buffer_chunk(buffer_size)
            coord_1 = buffer_chunk.loc(axis=1)[bodypart1, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
            coord_2 = buffer_chunk.loc(axis=1)[bodypart2, ['x', 'y', 'z']].droplevel('bodyparts', axis=1)
        else:
            stsw_mask = self.data_chunk.loc(axis=1)[self.stepping_limb, 'SwSt'] == step_phase
            coord_1 = self.data_chunk.loc(axis=1)[bodypart1, ['x', 'y', 'z']][stsw_mask].droplevel('bodyparts', axis=1)
            coord_2 = self.data_chunk.loc(axis=1)[bodypart2, ['x', 'y', 'z']][stsw_mask].droplevel('bodyparts', axis=1)

        # #Calculate the vectors from body1 to body2
        #
        # v1 = coord_1 - reference_axis
        # v2 = coord_2 - reference_axis
        #
        # v1_on_plane = v1 - np.dot(v1, reference_axis) * reference_axis
        # v2_on_plane = v2 - np.dot(v2, reference_axis) * reference_axis
        #
        # plane_normal = np.cross(v1_on_plane, v2_on_plane)
        # angle_sign = np.sign(np.dot(plane_normal, reference_axis)) #####!!!
        #
        # angle_radians = np.arctan2(angle_sign * np.linalg.norm(np.cross(v1_on_plane, v2_on_plane)), np.dot(v1_on_plane, v2_on_plane))
        # # Convert angle to degrees and ensure it is in the 0-360 range
        # angle_degrees = np.degrees(angle_radians) % 360
        # # return angle_degrees

        vectors_body1_to_body2 = coord_2 - coord_1

        # start morio
        # sign_angle = np.sign(np.cross(vectors_body1_to_body2, reference_axis))
        # sign_angle = sign_angle[:,1] * sign_angle[:,2]
        # end morio

        # Calculate the dot product between the vectors and the reference axis
        dot_products = np.dot(vectors_body1_to_body2, reference_axis)
        # Calculate the magnitudes of the vectors
        magnitudes_body1_to_body2 = np.linalg.norm(vectors_body1_to_body2, axis=1)
        magnitude_reference_axis = np.linalg.norm(reference_axis)
        # Calculate the cosine of the reach angle
        cosine_reach_angle = dot_products / (magnitudes_body1_to_body2 * magnitude_reference_axis)
        # Calculate the reach angle in radians
        angle_radians = np.arcsin(cosine_reach_angle)  # * sign_angle
        # Convert the reach angle to degrees
        angle_degrees = np.degrees(angle_radians) + 90

        if all_vals:
            return angle_degrees.droplevel(['Run', 'RunStage'], axis=0)
        else:
            return angle_degrees.mean()


########################################################################
########################################################################
class RunMeasures(CalculateMeasuresByStride):
    """
    calc_obj = CalculateMeasuresByStride(XYZw, con, mouseID, r, stride_start, stride_end, stepping_limb)
    run_measures = RunMeasures(measures_dict, calc_obj)
    results = run_measures.run()
    """
    def __init__(self, measures, calc_obj, buffer_size, stride):
        super().__init__(calc_obj.XYZw, calc_obj.mouseID, calc_obj.r,
                         calc_obj.stride_start, calc_obj.stride_end, calc_obj.stepping_limb, calc_obj.conditions)  # Initialize parent class with the provided arguments
        self.measures = measures
        self.buffer_size = buffer_size
        self.stride = stride

    def setup_df(self, datatype, measures):
        col_names = []
        for function in measures[datatype].keys():
            if any(measures[datatype][function]):
                if function != 'signed_angle':
                    for param in itertools.product(*measures[datatype][function].values()):
                        param_names = list(measures[datatype][function].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                        col_names.append((function, formatted_params))
                else:
                    for combo in measures[datatype]['signed_angle'].keys():
                        col_names.append((function, combo))
            else:
                col_names.append((function, 'no_param'))

        col_names_trimmed = []
        for c in col_names:
            if np.logical_and('full_stride:True' in c[1], 'step_phase:None' not in c[1]):
                pass
            elif np.logical_and('full_stride:False' in c[1], 'step_phase:None' in c[1]):
                pass
            else:
                col_names_trimmed.append(c)

        buffered_idx = self.get_buffer_chunk(self.buffer_size).index.get_level_values(level='FrameIdx')
        ## add in new index with either 'buffer' or 'stride' based on whether that frame is in the buffer or between stride start and end
        buffer_mask = np.logical_and(buffered_idx >= self.stride_start, buffered_idx <= self.stride_end)
        idx_type = np.where(buffer_mask, 'stride', 'buffer')


        if datatype == 'single_val_measure_list':
            vals = pd.DataFrame(index=[0], columns=pd.MultiIndex.from_tuples(col_names_trimmed, names=['Measure', 'Params']))
        elif datatype == 'multi_val_measure_list':
            mult_index = pd.MultiIndex.from_arrays([buffered_idx, idx_type], names=['FrameIdx', 'Buffer'])
            vals = pd.DataFrame(index=mult_index, columns=pd.MultiIndex.from_tuples(col_names_trimmed, names=['Measure', 'Params']))
        return vals

    def run_calculations(self, datatype, measures):
        vals = self.setup_df(datatype,measures)

        for function in measures[datatype].keys():
            if any(measures[datatype][function]):
                if function != 'signed_angle':
                    for param in itertools.product(*measures[datatype][function].values()):
                        param_names = list(measures[datatype][function].keys())
                        formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))

                        if np.logical_and('full_stride:True' in formatted_params, 'step_phase:None' not in formatted_params):
                            pass
                        elif np.logical_and('full_stride:False' in formatted_params, 'step_phase:None' in formatted_params):
                            pass
                        else:
                            result = getattr(self, function)(*param)
                            if datatype == 'single_val_measure_list':
                                vals.loc(axis=1)[(function, formatted_params)] = result
                            elif datatype == 'multi_val_measure_list':
                                # idx_mask = vals.index.get_level_values(level='Buffered_idx').isin(result.index)
                                # full_idx = vals.index[idx_mask]
                                vals.loc[result.index, (function, formatted_params)] = result.values

                else:
                    for combo in measures[datatype]['signed_angle'].keys():
                        result = getattr(self, function)(*measures[datatype][function][combo])
                        if datatype == 'single_val_measure_list':
                            vals.loc(axis=1)[(function, combo)] = result
                        elif datatype == 'multi_val_measure_list':
                            vals.loc[result.index, (function, combo)] = result.values

            else:
                # when no parameters required
                result = getattr(self, function)()
                if datatype == 'single_val_measure_list':
                    vals.loc(axis=1)[(function, 'no_param')] = result
                elif datatype == 'multi_val_measure_list':
                    vals.loc[result.index, (function, 'no_param')] = result.values

        return vals

    def add_single_idx(self, data):
        single_idx = pd.MultiIndex.from_tuples([(self.mouseID, int(self.r), self.stride)],
                                               names=['MouseID', 'Run', 'Stride'])
        data.set_index(single_idx, inplace=True)
        return data

    def add_multi_idx(selfself, data, single_data):
        single_idx = single_data.index
        data_idx = [data.index.get_level_values(level='FrameIdx'), data.index.get_level_values(level='Buffer')]
        multi_index_tuples = [(a[0], a[1], a[2], data_idx[0][b], data_idx[1][b]) for a in single_idx for b in np.arange(len(data_idx[0]))]
        multi_index = pd.MultiIndex.from_tuples(multi_index_tuples,
                                                names=['MouseID', 'Run', 'Stride', 'FrameIdx', 'Buffer'])
        data.set_index(multi_index, inplace=True)
        return data

    def get_all_results(self):
        single_val = self.run_calculations('single_val_measure_list', self.measures)
        multi_val = self.run_calculations('multi_val_measure_list', self.measures)

        # add in multi indexes
        single_val_indexed = self.add_single_idx(single_val)
        mult_val_indexed = self.add_multi_idx(multi_val,single_val_indexed)

        return single_val_indexed, mult_val_indexed
