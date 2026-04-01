"""Configuration and constants for the multi-camera labelling tool."""
from helpers.config import MAIN_DATA_ROOT

# USER SETTINGS
dir = MAIN_DATA_ROOT + "analysis/DLC_DualBelt/Manual_Labelling"

# Paths
DEFAULT_CALIBRATION_FILE_PATH = "%s/CameraCalibration/default_calibration_labels.csv" %(dir)
CALIBRATION_SAVE_PATH_TEMPLATE = "%s/CameraCalibration/{video_name}/calibration_labels.csv" %(dir)
FRAME_SAVE_PATH_TEMPLATE = {
    "side": "%s/Side/{video_name}" %(dir),
    "front": "%s/Front/{video_name}" %(dir),
    "overhead": "%s/Overhead/{video_name}" %(dir),
}
LABEL_SAVE_PATH_TEMPLATE = {
    "side": "%s/Side/{video_name}" %(dir),
    "front": "%s/Front/{video_name}" %(dir),
    "overhead": "%s/Overhead/{video_name}" %(dir),
}


# Labels
CALIBRATION_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR"]
BODY_PART_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR",
                    "Nose", "EarL", "EarR", "Back1", "Back2", "Back3", "Back4", "Back5", "Back6", "Back7", "Back8",
                    "Back9", "Back10", "Back11", "Back12", "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6",
                    "Tail7", "Tail8", "Tail9", "Tail10", "Tail11", "Tail12",
                    "ForepawToeR", "ForepawKnuckleR", "ForepawAnkleR", "ForepawKneeR",
                    "ForepawToeL", "ForepawKnuckleL", "ForepawAnkleL", "ForepawKneeL",
                    "HindpawToeR", "HindpawKnuckleR", "HindpawAnkleR", "HindpawKneeR",
                    "HindpawToeL", "HindpawKnuckleL", "HindpawAnkleL", "HindpawKneeL"]
# OPTIMIZATION_REFERENCE_LABELS = ['Nose', 'HindpawToeR', 'HindpawToeL', 'HindpawAnkleR', 'HindpawAnkleL', 'Back1', 'Tail1', 'Tail12']
# REFERENCE_LABEL_WEIGHTS = {
#     'Nose': 1.0,
#     'HindpawToeR': 1.0,
#     'HindpawToeL': 1.0,
#     'HindpawAnkleR': 1.0,
#     'HindpawAnkleL': 1.0,
#     'Back1': 1.0,
#     'Tail1': 1.0,
#     'Tail12': 1.0,
# }
OPTIMIZATION_REFERENCE_LABELS = ['Nose', 'EarL', 'EarR', 'ForepawToeR', 'ForepawToeL', 'HindpawToeR',
                                 'HindpawKnuckleR', 'Back1', 'Back3', 'Back6', 'Back9', 'Tail1', 'Tail6', 'Tail12',
                                 'StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR', 'TransitionL', 'Door']
REFERENCE_LABEL_WEIGHTS = {
    'side': {
        'Nose': 2.0,
        'EarL': 0.5,
        'EarR': 0.5,
        'ForepawToeR': 1.0,
        'ForepawToeL': 1.0,
        'HindpawToeR': 1.0,
        'HindpawKnuckleR': 1,
        'Back1': 1.0,
        'Back3': 1.0,
        'Back6': 1.0,
        'Back9': 1.0,
        'Tail1': 5.0,
        'Tail6': 1.0,
        'Tail12': 5.0,
        'StartPlatR': 0.5,
        'StepR': 0.5,
        'StartPlatL': 0.5,
        'StepL': 0.5,
        'TransitionR': 0.5,
        'TransitionL': 0.5,
        'Door': 10.0,
    },
    'front': {
        'Nose': 2.0,
        'EarL': 0.5,
        'EarR': 0.5,
        'ForepawToeR': 1.0,
        'ForepawToeL': 1.0,
        'HindpawToeR': 1.0,
        'HindpawKnuckleR': 1,
        'Back1': 1.0,
        'Back3': 1.0,
        'Back6': 1.0,
        'Back9': 1.0,
        'Tail1': 5.0,
        'Tail6': 1.0,
        'Tail12': 5.0,
        'StartPlatR': 0.5,
        'StepR': 0.5,
        'StartPlatL': 0.5,
        'StepL': 0.5,
        'TransitionR': 0.5,
        'TransitionL': 0.5,
        'Door': 10.0,
    },
    'overhead': {
        'Nose': 2.0,
        'EarL': 0.5,
        'EarR': 0.5,
        'ForepawToeR': 1.0,
        'ForepawToeL': 1.0,
        'HindpawToeR': 1.0,
        'HindpawKnuckleR': 1,
        'Back1': 1.0,
        'Back3': 1.0,
        'Back6': 1.0,
        'Back9': 1.0,
        'Tail1': 5.0,
        'Tail6': 1.0,
        'Tail12': 5.0,
        'StartPlatR': 0.5,
        'StepR': 0.5,
        'StartPlatL': 0.5,
        'StepL': 0.5,
        'TransitionR': 0.5,
        'TransitionL': 0.5,
        'Door': 10.0,}
}

# Marker Size
DEFAULT_MARKER_SIZE = 1
MIN_MARKER_SIZE = 0.1
MAX_MARKER_SIZE = 5
MARKER_SIZE_STEP = 0.1

# Contrast and Brightness
DEFAULT_CONTRAST = 1.0
DEFAULT_BRIGHTNESS = 1.0
MIN_CONTRAST = 0.5
MAX_CONTRAST = 3.0
CONTRAST_STEP = 0.1
MIN_BRIGHTNESS = 0.5
MAX_BRIGHTNESS = 3.0
BRIGHTNESS_STEP = 0.1

# Coordinate data format
SCORER = "Holly"
