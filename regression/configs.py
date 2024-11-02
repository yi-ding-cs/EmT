import os

emotion = ["Valence"]

### For preprocessing
### If tagged with "# Check this", then it's adjustable, otherwise leave it alone.
config = {

    "extract_class_label": 1,
    "extract_continuous_label": 1,

    "extract_eeg": 1,
    "eeg_folder": "eeg",
    "eeg_config": {
        "sampling_frequency": 256,
        "window_sec": 2,  # Check this
        "hop_sec": 0.25, # Check this
        "buffer_sec": 5, # Check this
        "num_electrodes": 32,
        "interest_bands": [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)], # Check this
        "f_trans_interest_bands": [(0.1, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], # Check this
        "channel_slice": {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)},
        "features": ["eeg_bandpower"],
        "filter_type": 'cheby2',   #cheby2, butter
        "filter_order": 4
    },

    "save_npy": 1,
    "npy_folder": "compacted_48",

    "dataset_name": "mahnob",
    "emotion_list": emotion,
    "root_directory": r"D:\DingYi\Dataset\MAHNOB-O", # Check this
    "output_root_directory": r"D:\DingYi\Dataset\MAHNOB-P",  # Check this
    "raw_data_folder": "Sessions",

    "multiplier": {
        "video": 16,
        "eeg_raw":1,
        "eeg_bandpower": 1,
        "eeg_DE": 1,
        "eeg_RP": 1,
        "eeg_Hjorth": 1,
        "continuous_label": 1,
    },

    "feature_dimension": {
        "eeg_raw": (16384,),
        "eeg_bandpower": (192,),
        "eeg_DE": (192,),
        "eeg_RP": (192,),
        "eeg_Hjorth": (96,),
        "continuous_label": (1,),
        "class_label": (1,),
    },

    "max_epoch": 30, # Check this
    "min_epoch": 0,

    "model_name": "2d1d",  # Check this ## No actual use but only a naming issue.

    "backbone": {
        "state_dict": "res50_ir_0.887",
        "mode": "ir",
    },


    "early_stopping": 30, # Check this
    "load_best_at_each_epoch": 0, # Check this
    "time_delay": 0, # Check this, Move the continuous label afterward for n data points. One data point = 0.25s.
    "metrics": ["rmse", "pcc", "ccc"],
    "save_plot": 0,
}
