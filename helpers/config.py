"""Centralized configuration for paths, experiment structure, mouse metadata, and feature definitions."""
import numpy as np

#################################### Paths ####################################
###############################################################################
MAIN_DATA_ROOT = r"H:\Dual-belt_APAs"
PLOTS_ROOT = r"H:\Characterisation_v2"
CEPH_ROOT = r"X:\hmorley\Dual-belt_APAs"

paths = {
        'filtereddata_folder': MAIN_DATA_ROOT + r"\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25",
        'plotting_destfolder': PLOTS_ROOT + r'\Jan25',
        'video_folder': CEPH_ROOT + r"\videos\Round_3",
        'local_video_folder': MAIN_DATA_ROOT + r"\videos\Round_3",
        'data_folder': MAIN_DATA_ROOT + r"\analysis\DLC_DualBelt\DualBelt_AnalysedFiles",
}

################################## Videos #####################################
###############################################################################

vidstuff = {
    'cams': ['side','front','overhead'],
    'scorers': {
        'side': "DLC_resnet50_DLC_DualBeltAug2shuffle1_1200000",
        'overhead': "DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000",
        'front': "DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000"
    }
}
fps = 247

pcutoff = 0.9

################################ Experiments ##################################
###############################################################################

expstuff = {
    'exp_chunks': {
        'RunStages': ['TrialStart', 'RunStart', 'Transition', 'RunEnd', 'TrialEnd'],
        'ExpPhases': ['Baseline', 'APA', 'Washout']
    },
    'condition_exp_lengths': {
        'APACharRuns': [10,20,10],
        'APAPerRuns': [10,20,10],  # N.B. sometimes uses APAChar values for perception runs
        'APAVmtRuns': [10,15,10]
    },
    'condition_exp_runs': {
        'APAChar': {
            'Repeats': { # 10, 20, 10
                'Baseline': np.arange(0, 10),
                'APA1': np.arange(10, 20),
                'APA2': np.arange(20, 30),
                'Washout': np.arange(30, 40)
            },
            'Extended': { # 10, 50, 50, 50
                'Baseline': np.arange(0, 10),
                'APA': np.arange(10, 110),
                'APA1': np.arange(10, 60),
                'APA2': np.arange(60, 110),
                'Washout': np.arange(110, 160),
                'Wash1': np.arange(110, 135),
                'Wash2': np.arange(135, 160)
            }
        }
    },
    'condition_exp_runs_basic': {
        'APAChar': {
            'Repeats': {  # 10, 20, 10
                'Baseline': np.arange(0, 10),
                'APA': np.arange(10, 30),
                'Washout': np.arange(30, 40)
            },
            'Extended': {  # 10, 50, 50, 50
                'Baseline': np.arange(0, 10),
                'APA': np.arange(10, 110),
                'Washout': np.arange(110, 160),
            }
        }
    },
    'setup': {
        'distancescm': [13,12,11.5,11.5], # goes 0: wall1-wall0,  1: wall2-wall1, 3: wall3-wall2, 4: wall4-wall3
        'transition_mm': 470, # mm
    },
    'speeds': {'Low': 6, 'Mid': 18, 'High': 30},
    'preprun_nos': {
        'preruns_CharLow': 2,
        'preruns_CharMidHigh': 5
    }
}

####################### Experimental Categories ###############################
# Maps experiment conditions to recording dates and mouse groups (A/B)
###############################################################################

exp_cats = {
    'APAChar_LowHigh': {
        'Repeats': {
            'Wash': {
                'Exp': {
                    'Day1': {'A': ['20230306'], 'B': ['20230309']},
                    'Day2': {'A': ['20230308'], 'B': ['20230312']},
                    'Day3': {'A': ['20230310'], 'B': ['20230314']},
                },
                'Washout': {
                    'Day1': {'A': ['20230307'], 'B': ['20230310']},
                    'Day2': {'A': ['20230309'], 'B': ['20230313']},
                    'Day3': {'A': ['20230312'], 'B': ['20230315']},
                }
            }
        },
        'Extended': {
            'Day1': {'A': ['20230316'], 'B': ['20230403']},
            'Day2': {'A': ['20230317'], 'B': ['20230404']},
            'Day3': {'A': ['20230318'], 'B': ['20230405']},
            'Day4': {'A': ['20230319'], 'B': ['20230406']}
        }
    },
    'APAChar_HighLow': {
        'Extended': {
            'Day1': {'A': ['20230325'], 'B': ['20230412']},
            'Day2': {'A': ['20230326'], 'B': ['20230413']},
            'Day3': {'A': ['20230327'], 'B': ['20230414']},
            'Day4': {'A': ['20230328'], 'B': ['20230415']}
        }
    },
    'APAChar_LowMid': {
        'Extended': {
            'Day1': {'A': ['20230407'], 'B': ['20230320']},
            'Day2': {'A': ['20230408'], 'B': ['20230321']},
            'Day3': {'A': ['20230409'], 'B': ['20230322']},
            'Day4': {'A': ['20230410'], 'B': ['20230323']}
        }
    },
    'PerceptionTest': {'A': ['20230411'], 'B': ['20230416']},
    'VMT_LowHigh': {
        'pd': {
            'Basic': {
                'Prep': {'A': ['20230420'], 'B': ['20230427']},
                'Experiment': {'A': ['20230421'], 'B': ['20230428']}
            },
            'Extreme': {
                'Prep': {'A': ['20230513'], 'B': ['20230514']},
                'Experiment': {'A': ['20230514'], 'B': ['20230515']}
            }
        },
        'ac': {
            'Basic': {
                'Prep': {'A': ['20230426'], 'B': ['20230503']},
                'Experiment': {'A': ['20230427'], 'B': ['20230504']}
            },
            'Extreme': {
                'Prep': {'A': ['20230515'], 'B': ['20230516']},
                'Experiment': {'A': ['20230516'], 'B': ['20230517']}
            }
        }
    },
    'VMT_HighLow': {
        'pd': {
            'Basic': {
                'Prep': {'A': ['20230504'], 'B': ['20230421']},
                'Experiment': {'A': ['20230505'], 'B': ['20230422']}
            }
        },
        'ac': {
            'Basic': {
                'Prep': {'A': ['20230428'], 'B': ['20230505']},
                'Experiment': {'A': ['20230429'], 'B': ['20230506']}
            }
        }
    }
}

########################### Apparatus Geometry ################################
###############################################################################

structural_stuff = {
    'belt_width': 53.5, # mm
    'belt_length_sideviewrange': 600, # mm
    'belt_length_sideviewend': 130 # mm
}
########################## Gait / Locomotion ##################################
###############################################################################

locostuff = {
    'swst_vals_2025': {
        'st': '1',
        'sw': '0'
    },
}
############################ Analysis Settings ################################
###############################################################################

settings = {
    'analysis_chunks': {
        'APA_lengthruns': 25,
        'after_lengthruns': 25,
        'literature_apa_length': 100  # ms
    }
}

############################### Logging #######################################
###############################################################################

logging = {
    'error': ['File','exp','speed','repeat_extend','exp_wash','day','vmt_type','vmt_level','prep','MouseID','RunNumber',
              'ErrorType','ErrorMessage'],
    'run_summary': ['File', 'exp', 'speed', 'repeat_extend', 'exp_wash','day','vmt_type', 'vmt_level', 'prep','MouseID',
                    'RegisteredRuns','RecordedRuns','MissingRuns','DroppedRunsPlaceholder','DroppedRunsCompletely', ]
}

########################## Mouse IDs & Anatomy ################################
###############################################################################

micestuff = {
    'mice_IDs': {
        'A': ['1035243', '1035244', '1035245', '1035246', '1035249', '1035250'],
        'B': ['1035297', '1035298', '1035299', '1035301', '1035302']
    },
    'mice_ID': [  # Legacy cohort IDs, used by Utils.checkFilenamesMouseID
        '1034976', '1034978', '1034979',
        '1034980', '1034982', '1034983'
    ],
    'mice_name': [
        'MNone', 'MR', 'MLR',
        'MNone', 'MR', 'MLR'
    ],
    'LR': {
        'ForepawToeL': 1,
        'ForepawToeR': 2,
        'ForepawL': 1,
        'ForepawR': 2,
        'ForepawL_slid': 1,
        'ForepawR_slid': 2,
    },
    'skeleton':[
        ('Nose', 'Back1'), ('EarL', 'Back1'), ('EarR', 'Back1'),
        ('Back1', 'Back2'), ('Back2', 'Back3'), ('Back3', 'Back4'), ('Back4', 'Back5'), ('Back5', 'Back6'),
        ('Back6', 'Back7'), ('Back7', 'Back8'), ('Back8', 'Back9'), ('Back9', 'Back10'), ('Back10', 'Back11'),
        ('Back11', 'Back12'), ('Back12', 'Tail1'),
        ('Tail1', 'Tail2'), ('Tail2', 'Tail3'), ('Tail3', 'Tail4'), ('Tail4', 'Tail5'), ('Tail5', 'Tail6'),
        ('Tail6', 'Tail7'), ('Tail7', 'Tail8'), ('Tail8', 'Tail9'), ('Tail9', 'Tail10'), ('Tail10', 'Tail11'),
        ('Tail11', 'Tail12'),
        ('Back3', 'ForepawKneeL'), ('Back3', 'ForepawKneeR'), ('Back9', 'HindpawKneeL'),('Back9', 'HindpawKneeR'),
        ('ForepawKneeL', 'ForepawAnkleL'), ('ForepawKneeR', 'ForepawAnkleR'),
        ('HindpawKneeL', 'HindpawAnkleL'), ('HindpawKneeR', 'HindpawAnkleR'),
        ('ForepawAnkleL', 'ForepawKnuckleL'), ('ForepawAnkleR', 'ForepawKnuckleR'),
        ('HindpawAnkleL', 'HindpawKnuckleL'), ('HindpawAnkleR', 'HindpawKnuckleR'),
        ('ForepawKnuckleL', 'ForepawToeL'), ('ForepawKnuckleR', 'ForepawToeR'),
        ('HindpawKnuckleL', 'HindpawToeL'), ('HindpawKnuckleR', 'HindpawToeR')
        ]
    ,
    'bodyparts': ['Nose', 'EarL', 'EarR',
                  'Back1', 'Back2', 'Back3','Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10','Back11', 'Back12',
                  'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5','Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12',
                  'ForepawToeL', 'ForepawToeR',
                  'ForepawKnuckleL', 'ForepawKnuckleR',
                  'ForepawAnkleL', 'ForepawAnkleR',
                  'ForepawKneeL', 'ForepawKneeR',
                  'HindpawToeL', 'HindpawToeR',
                  'HindpawKnuckleL', 'HindpawKnuckleR',
                  'HindpawAnkleL', 'HindpawAnkleR',
                  'HindpawKneeL', 'HindpawKneeR']
}

########################## Labelling Definitions ##############################
###############################################################################

label_list_World = ['Door','StartPlatL','StartPlatR','TransitionL','TransitionR', 'StepR', 'StepL',
                    'Nose','EarL','EarR',
                    'Back1','Back2','Back3','Back4','Back5','Back6','Back7','Back8','Back9','Back10','Back11','Back12',
                    'Tail1','Tail2','Tail3','Tail4','Tail5','Tail6','Tail7','Tail8','Tail9','Tail10','Tail11','Tail12',
                    'ForepawToeL','ForepawToeR','ForepawKnuckleL','ForepawKnuckleR','ForepawAnkleL','ForepawAnkleR','ForepawKneeL','ForepawKneeR',
                    'HindpawToeL','HindpawToeR', 'HindpawKnuckleL','HindpawKnuckleR','HindpawAnkleL','HindpawAnkleR','HindpawKneeL','HindpawKneeR']

label_list = {
    'sideXfront': ['ForepawToeR', 'ForepawAnkleR', 'HindpawToeR', 'HindpawAnkleR','ForepawToeL', 'ForepawAnkleL',
                   'HindpawToeL', 'HindpawAnkleL'],
    'sideXoverhead': ['Nose', 'EarL', 'EarR', 'Back1', 'Back2', 'Back3','Back4', 'Back5', 'Back6', 'Back7', 'Back8',
                      'Back9', 'Back10','Back11', 'Back12', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5','Tail6',
                      'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12', 'StartPlatR', 'StartPlatL', 'TransitionR', 'TransitionL']
}

########################## Feature Measure Lists ##############################
###############################################################################

measure_type_names = ['single_kinematics_runXstride', 'multi_kinematics_runXstride', 'behaviour_run']

def measures_list(buffer): ## add in displacement??
    measures = {
        'multi_val_measure_list' : {
            'instantaneous_swing_velocity': {
                'speed_correct': [True,False],
                'xyz': ['x','y','z'],
                'smooth': [False]
            },
            'net_displacement_rel': {
                'coord': ['x','y','z'],
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','FrontIpsi','FrontContra','HindIpsi','HindContra'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'traj': {
                'bodypart': ['Nose','Back1','Back6','Back12','Tail1','Tail6','Tail12','FrontIpsi','FrontContra','HindIpsi','HindContra'],
                'coord': ['x','y','z'],
                'step_phase': [None],
                'full_stride': [True],
                'speed_correct': [True, False],
                'aligned': [True], # False
                'buffer_size': [0, buffer]
            },
            'body_length': {
                'bodyparts': [['Back1','Back12']],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'back_height': {
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'tail_height': {
                'tail_label': ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10','Tail11', 'Tail12'],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'back_skew': {
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
            },
            'limb_rel_to_body': {
                'time': [None],
                'step_phase': [None],
                'all_vals': [True],
                'full_stride': [True],
                'buffer_size': [0, buffer]
             },
        },
        'single_val_measure_list': {
            'stride_duration': [],
            'stance_duration': [],
            'swing_duration': [],
            'cadence': [],
            'duty_factor': [],
            'brake_prop_duration': {
                'type': ['brake', 'propulsion'],
            },
            'walking_speed': {
                'bodypart': ['Back6','Tail1'],
                'speed_correct': [True]
            },
            'swing_velocity': {
                'speed_correct': [True]
            },
            'stride_length': {
                'speed_correct': [True]
            },
            'stance_phase': {
                'stance_limb': ['contra_front', 'contra_hind', 'ipsi_hind']
            },
            'nose_tail_phase': {
                'bodypart': ['Nose', 'Tail1'],
                'frontback': ['front', 'hind'],
                'coord': ['x', 'y', 'z'],
            },
            'double_support': {
                'type': ['frontonly', 'homolateral', 'diagonal'],
            },
            'triple_support': {
                'mode': ['any', 'front_hind'],
            },
            'quadruple_support': [],
            'net_displacement_rel': {
                'coord': ['x','y','z'],
                'bodypart': ['Nose','Back1','Back6','Tail6','Tail12','FrontIpsi','FrontContra','HindIpsi','HindContra'],
                'step_phase': ['0', '1', None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0]
            },
            'distance_to_transition': {
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'distance_from_midline': {
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'distance_from_belt_surface': {
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'excursion': {
                'bodypart': ['Nose','Back1','Back6','Tail1','Tail6','Tail12','FrontIpsi','FrontContra','HindIpsi','HindContra'],
                'coord': ['x','y','z'],
                'buffer_size': [0]
            },
            'coo_xyz': {
                'xyz': ['x','y','z']
            },
            'coo_euclidean': [],
            'bos_stancestart': {
                'ref_or_contr': ['ref','contr'],
                'y_or_euc': ['y','euc']
            },
            'ptp_amplitude_stride': {
                'bodypart': ['Tail1','Back6']
            },
            'body_length': {
                'bodyparts': [['Back1','Back12']],
                'step_phase': ['0', '1', None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0]
            },
            'back_height': {
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12'],
                'step_phase': ['0', '1', None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0]
            },
            'tail_height': {
                'tail_label': ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12'],
                'step_phase': ['0', '1', None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0]
            },
            'back_skew': {
                'step_phase': ['0', '1', None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0]
            },
            'limb_rel_to_body': {
                'time': ['start','end'],
                'step_phase': ['0', '1', None],
                'all_vals': [False],
                'full_stride': [True, False],
                'buffer_size': [0]
            },
            'signed_angle': {
                'ToeKnuckle_ipsi_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeKnuckle_contra_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeAnkle_ipsi_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['FrontAnkleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeAnkle_contra_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontAnkleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'Back1Back12_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'Tail1Tail12_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'NoseBack1_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'Back1Back12_overhead_xref_swing_mean': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'Tail1Tail12_overhead_xref_swing_mean': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'NoseBack1_overhead_xref_swing_mean': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeKnuckle_ipsi_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'ToeKnuckle_contra_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'ToeAnkle_ipsi_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['FrontAnkleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'ToeAnkle_contra_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontAnkleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'Back1Back12_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'Tail1Tail12_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'NoseBack1_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'Back1Back12_overhead_xref_swing_peak': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'Tail1Tail12_overhead_xref_swing_peak': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'NoseBack1_overhead_xref_swing_peak': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
            }
        }
    }
    return measures

# Manually curated subset of measures for interpretable analysis
measures_list_manual_reduction = {
            'stride_duration': [],
            'cadence': [],
            'duty_factor': [],
            'brake_prop_duration': {
                'type': ['brake', 'propulsion'],
            },
            'walking_speed': {
                'bodypart': ['Tail1'],
                'speed_correct': [True]
            },
            'swing_velocity': {
                'speed_correct': [True]
            },
            'stride_length': {
                'speed_correct': [True]
            },
            'stance_phase': {
                'stance_limb': ['contra_front', 'contra_hind', 'ipsi_hind']
            },
            'nose_tail_phase': {
                'bodypart': ['Nose', 'Tail1'],
                'frontback': ['front', 'hind'],
                'coord': ['x', 'y', 'z'],
            },
            'double_support': {
                'type': ['frontonly'],#, 'homolateral', 'diagonal'],
            },
            'triple_support': {
                'mode': ['any'],#, 'front_hind'],
            },
            'quadruple_support': [],
            'distance_to_transition': {
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'distance_from_midline': {
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'coo_euclidean': [],
            'bos_stancestart': {
                'ref_or_contr': ['ref','contr'],
                'y_or_euc': ['y','euc']
            },
            'ptp_amplitude_stride': {
                'bodypart': ['Tail1','Back6']
            },
            'body_length': {
                'bodyparts': [['Back1','Back12']],
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'back_height': {
                'back_label': ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12'],
                'step_phase': [None],
                'all_vals': [False],
                'full_stride': [True],
                'buffer_size': [0]
            },
            'tail_height': {
                'tail_label': ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12'],
                'step_phase': [None],
                'all_vals': [False],
                'full_stride': [True],
                'buffer_size': [0]
            },
            'back_skew': {
                'step_phase': ['0', '1'],
                'all_vals': [False],
                'full_stride': [False],
                'buffer_size': [0]
            },
            'signed_angle': {
                'ToeKnuckle_ipsi_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeKnuckle_contra_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeAnkle_ipsi_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['FrontAnkleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeAnkle_contra_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontAnkleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'Back1Back12_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'Tail1Tail12_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'NoseBack1_side_zref_swing_mean': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'Back1Back12_overhead_xref_swing_mean': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'mean',0],
                'Tail1Tail12_overhead_xref_swing_mean': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'NoseBack1_overhead_xref_swing_mean': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'mean', 0],
                'ToeKnuckle_ipsi_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'ToeKnuckle_contra_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontKnuckleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'ToeAnkle_ipsi_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['FrontAnkleIpsi', 'FrontToeIpsi'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'ToeAnkle_contra_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]),['FrontAnkleContra', 'FrontToeContra'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'Back1Back12_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'Tail1Tail12_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'NoseBack1_side_zref_swing_peak': [np.array([0, 0, 1]), np.array([0, 1, 0]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'Back1Back12_overhead_xref_swing_peak': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Back1', 'Back12'],locostuff['swst_vals_2025']['sw'],False,False,'peak',0],
                'Tail1Tail12_overhead_xref_swing_peak': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Tail1', 'Tail12'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
                'NoseBack1_overhead_xref_swing_peak': [np.array([1, 0, 0]), np.array([0, 0, 1]), ['Nose', 'Back1'],locostuff['swst_vals_2025']['sw'],False,False,'peak', 0],
            }
}

####################### Manual Feature Clusters ###############################
###############################################################################

manual_clusters = {
    'cluster_values': {
        'Gait timing': 1,
        'Gait stability': 2,
        'Positioning on belt': 3,
        'Limb coordination': 4,
        'Limb-Body coordination': 5,
        'Body coordination/positioning': 6,
    },
    'cluster_mapping': {
        # Gait timing: cluster 1
        'stride_duration': 1,
        'stride_length|speed_correct:True': 1,
        'cadence': 1,
        'brake_prop_duration|type:brake': 1,
        'brake_prop_duration|type:propulsion': 1,
        'walking_speed|bodypart:Tail1, speed_correct:True': 1,
        'swing_velocity|speed_correct:True': 1,

        # Gait stability: cluster 2
        'duty_factor': 2,
        'average_support_val': 2,
        'bos_stancestart|ref_or_contr:ref, y_or_euc:y': 2,
        'bos_stancestart|ref_or_contr:ref, y_or_euc:euc': 2,
        'bos_stancestart|ref_or_contr:contr, y_or_euc:y': 2,
        'bos_stancestart|ref_or_contr:contr, y_or_euc:euc': 2,
        'coo_euclidean': 2,

        # Positioning on belt: cluster 3
        'distance_to_transition|step_phase:0, all_vals:False, full_stride:False, buffer_size:0': 3,
        'distance_to_transition|step_phase:1, all_vals:False, full_stride:False, buffer_size:0': 3,
        'distance_from_midline|step_phase:0, all_vals:False, full_stride:False, buffer_size:0': 3,
        'distance_from_midline|step_phase:1, all_vals:False, full_stride:False, buffer_size:0': 3,

        # Limb coordination: cluster 4
        'stance_phase|stance_limb:contra_front': 4,
        'stance_phase|stance_limb:contra_hind': 4,
        'stance_phase|stance_limb:ipsi_hind': 4,

        # Limb-Body coordination: cluster 5
        'nose_tail_phase|bodypart:Nose, frontback:front, coord:x': 5,
        'nose_tail_phase|bodypart:Tail1, frontback:front, coord:x': 5,
        'nose_tail_phase|bodypart:Nose, frontback:front, coord:y': 5,
        'nose_tail_phase|bodypart:Tail1, frontback:front, coord:y': 5,
        'nose_tail_phase|bodypart:Nose, frontback:front, coord:z': 5,
        'nose_tail_phase|bodypart:Tail1, frontback:front, coord:z': 5,
        'nose_tail_phase|bodypart:Nose, frontback:hind, coord:x': 5,
        'nose_tail_phase|bodypart:Tail1, frontback:hind, coord:x': 5,
        'nose_tail_phase|bodypart:Nose, frontback:hind, coord:y': 5,
        'nose_tail_phase|bodypart:Tail1, frontback:hind, coord:y': 5,
        'nose_tail_phase|bodypart:Nose, frontback:hind, coord:z': 5,
        'nose_tail_phase|bodypart:Tail1, frontback:hind, coord:z': 5,

        # Body coordination/positioning: cluster 6
        'ptp_amplitude_stride|bodypart:Tail1': 6,
        'ptp_amplitude_stride|bodypart:Back6': 6,
        "body_length|bodyparts:['Back1', 'Back12'], step_phase:0, all_vals:False, full_stride:False, buffer_size:0": 6,
        "body_length|bodyparts:['Back1', 'Back12'], step_phase:1, all_vals:False, full_stride:False, buffer_size:0": 6,
        'back_height_mean': 6,
        'tail_height_mean': 6,
        'back_skew|step_phase:0, all_vals:False, full_stride:False, buffer_size:0': 6,
        'back_skew|step_phase:1, all_vals:False, full_stride:False, buffer_size:0': 6,
        'signed_angle|ToeKnuckle_ipsi_side_zref_swing_mean': 6,
        'signed_angle|ToeKnuckle_ipsi_side_zref_swing_peak': 6,
        'signed_angle|ToeKnuckle_contra_side_zref_swing_mean': 6,
        'signed_angle|ToeKnuckle_contra_side_zref_swing_peak': 6,
        'signed_angle|ToeAnkle_ipsi_side_zref_swing_mean': 6,
        'signed_angle|ToeAnkle_ipsi_side_zref_swing_peak': 6,
        'signed_angle|ToeAnkle_contra_side_zref_swing_mean': 6,
        'signed_angle|ToeAnkle_contra_side_zref_swing_peak': 6,
        'signed_angle|Back1Back12_side_zref_swing_mean': 6,
        'signed_angle|Back1Back12_side_zref_swing_peak': 6,
        'signed_angle|Tail1Tail12_side_zref_swing_mean': 6,
        'signed_angle|Tail1Tail12_side_zref_swing_peak': 6,
        'signed_angle|NoseBack1_side_zref_swing_mean': 6,
        'signed_angle|NoseBack1_side_zref_swing_peak': 6,
        'signed_angle|Back1Back12_overhead_xref_swing_mean': 6,
        'signed_angle|Back1Back12_overhead_xref_swing_peak': 6,
        'signed_angle|Tail1Tail12_overhead_xref_swing_mean': 6,
        'signed_angle|Tail1Tail12_overhead_xref_swing_peak': 6,
        'signed_angle|NoseBack1_overhead_xref_swing_mean': 6,
        'signed_angle|NoseBack1_overhead_xref_swing_peak': 6
    }
}

##################### Display Names for Plotting ##############################
###############################################################################

short_names = {
        'stride_duration': 'Stride duration',
        'stride_length|speed_correct:True': 'Stride length',
        'cadence': 'Cadence',
        'brake_prop_duration|type:brake': 'Brake duration',
        'brake_prop_duration|type:propulsion': 'Propulsion duration',
        'walking_speed|bodypart:Tail1, speed_correct:True': 'Walking speed',
        'swing_velocity|speed_correct:True': 'Swing velocity',

        # Gait stability: cluster 2
        'duty_factor': 'Duty factor',
        'average_support_val': 'Average support value',
        'bos_stancestart|ref_or_contr:ref, y_or_euc:y': 'BOS (Ref st) - y',
        'bos_stancestart|ref_or_contr:ref, y_or_euc:euc': 'BOS (Ref st) - euclidean',
        'bos_stancestart|ref_or_contr:contr, y_or_euc:y': 'BOS (Contra st) - y',
        'bos_stancestart|ref_or_contr:contr, y_or_euc:euc': 'BOS (Contra st) - euclidean',
        'coo_euclidean': 'Center of oscillation - euclidean distance',

        # Positioning on belt: cluster 3
        'distance_to_transition|step_phase:0, all_vals:False, full_stride:False, buffer_size:0': 'Distance to transition (sw)',
        'distance_to_transition|step_phase:1, all_vals:False, full_stride:False, buffer_size:0': 'Distance to transition (st)',
        'distance_from_midline|step_phase:0, all_vals:False, full_stride:False, buffer_size:0': 'Distance from midline (sw)',
        'distance_from_midline|step_phase:1, all_vals:False, full_stride:False, buffer_size:0': 'Distance from midline (st)',

        # Limb coordination: cluster 4
        'stance_phase|stance_limb:contra_front': 'Relative st phase - contra front limb',
        'stance_phase|stance_limb:contra_hind': 'Relative st phase - contra hind limb',
        'stance_phase|stance_limb:ipsi_hind': 'Relative st phase - ipsi hind limb',

        # Limb-Body coordination: cluster 5
        'nose_tail_phase|bodypart:Nose, frontback:front, coord:x': 'Forepaw nose phase (x)',
        'nose_tail_phase|bodypart:Tail1, frontback:front, coord:x': 'Forepaw tail phase (x)',
        'nose_tail_phase|bodypart:Nose, frontback:front, coord:y': 'Forepaw nose phase (y)',
        'nose_tail_phase|bodypart:Tail1, frontback:front, coord:y': 'Forepaw tail phase (y)',
        'nose_tail_phase|bodypart:Nose, frontback:front, coord:z': 'Forepaw nose phase (z)',
        'nose_tail_phase|bodypart:Tail1, frontback:front, coord:z': 'Forepaw tail phase (z)',
        'nose_tail_phase|bodypart:Nose, frontback:hind, coord:x':  'Hindpaw nose phase (x)',
        'nose_tail_phase|bodypart:Tail1, frontback:hind, coord:x': 'Hindpaw tail phase (x)',
        'nose_tail_phase|bodypart:Nose, frontback:hind, coord:y': 'Hindpaw nose phase (y)',
        'nose_tail_phase|bodypart:Tail1, frontback:hind, coord:y': 'Hindpaw tail phase (y)',
        'nose_tail_phase|bodypart:Nose, frontback:hind, coord:z': 'Hindpaw nose phase (z)',
        'nose_tail_phase|bodypart:Tail1, frontback:hind, coord:z': 'Hindpaw tail phase (z)',

        # Body coordination/positioning: cluster 6
        'ptp_amplitude_stride|bodypart:Tail1': 'Peak to peak amp (Tail1)',
        'ptp_amplitude_stride|bodypart:Back6': 'Peak to peak amp (Back6)',
        "body_length|bodyparts:['Back1', 'Back12'], step_phase:0, all_vals:False, full_stride:False, buffer_size:0": 'Body length (sw)',
        "body_length|bodyparts:['Back1', 'Back12'], step_phase:1, all_vals:False, full_stride:False, buffer_size:0": 'Body length (st)',
        'back_height_mean': 'Back height mean',
        'tail_height_mean': 'Tail height mean',
        'back_skew|step_phase:0, all_vals:False, full_stride:False, buffer_size:0': 'Back skew (sw)',
        'back_skew|step_phase:1, all_vals:False, full_stride:False, buffer_size:0': 'Back skew (st)',
        'signed_angle|ToeKnuckle_ipsi_side_zref_swing_mean': 'Ipsi Toe-Knuckle pitch (mean)',
        'signed_angle|ToeKnuckle_ipsi_side_zref_swing_peak': 'Ipsi Toe-Knuckle pitch (peak)',
        'signed_angle|ToeKnuckle_contra_side_zref_swing_mean': 'Contra Toe-Knuckle pitch (mean)',
        'signed_angle|ToeKnuckle_contra_side_zref_swing_peak': 'Contra Toe-Knuckle pitch (peak)',
        'signed_angle|ToeAnkle_ipsi_side_zref_swing_mean': 'Ipsi Toe-Ankle pitch (mean)',
        'signed_angle|ToeAnkle_ipsi_side_zref_swing_peak': 'Ipsi Toe-Ankle pitch (peak)',
        'signed_angle|ToeAnkle_contra_side_zref_swing_mean': 'Contra Toe-Ankle pitch (mean)',
        'signed_angle|ToeAnkle_contra_side_zref_swing_peak': 'Contra Toe-Ankle pitch (peak)',
        'signed_angle|Back1Back12_side_zref_swing_mean': 'Back pitch (mean)',
        'signed_angle|Back1Back12_side_zref_swing_peak': 'Back pitch (peak)',
        'signed_angle|Tail1Tail12_side_zref_swing_mean': 'Tail pitch (mean)',
        'signed_angle|Tail1Tail12_side_zref_swing_peak': 'Tail pitch (peak)',
        'signed_angle|NoseBack1_side_zref_swing_mean': 'Head pitch (mean)',
        'signed_angle|NoseBack1_side_zref_swing_peak': 'Head pitch (peak)',
        'signed_angle|Back1Back12_overhead_xref_swing_mean': 'Back yaw (mean)',
        'signed_angle|Back1Back12_overhead_xref_swing_peak': 'Back yaw (peak)',
        'signed_angle|Tail1Tail12_overhead_xref_swing_mean': 'Tail yaw (mean)',
        'signed_angle|Tail1Tail12_overhead_xref_swing_peak': 'Tail yaw (peak)',
        'signed_angle|NoseBack1_overhead_xref_swing_mean': 'Head yaw (mean)',
        'signed_angle|NoseBack1_overhead_xref_swing_peak': 'Head yaw (peak)',
}
