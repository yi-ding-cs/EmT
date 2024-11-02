# from base.video import change_video_fps, combine_annotated_clips, OpenFaceController
# from base.audio import convert_video_to_wav, change_wav_frequency, extract_mfcc, extract_egemaps
# from base.speech import extract_transcript, add_punctuation, extract_word_embedding, align_word_embedding
from base.utils import ensure_dir, get_filename_from_a_folder_given_extension, save_to_pickle

import os

from operator import itemgetter
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image


class GenericDataPreprocessing(object):
    def __init__(self, config):

        self.config = config
        self.dataset_info = self.init_dataset_info()

        if "extract_continuous_label" in config and config['extract_continuous_label']:
            self.extract_continuous_label = config['extract_continuous_label']

        if "extract_class_label" in config and config['extract_class_label']:
            self.extract_class_label = config['extract_class_label']

        if "extract_eeg" in config and config['extract_eeg']:
            from base.eeg import GenericEegController
            self.extract_eeg = config['extract_eeg']
            self.eeg_folder = config['eeg_folder']

        self.per_trial_info = {}

    def get_output_root_directory(self):
        return self.config['output_root_directory']

    def prepare_data(self):

        for idx in tqdm(self.per_trial_info.keys(), total=len(self.per_trial_info.keys())):

            self.per_trial_info[idx]['processing_record'] = {}
            get_output_filename_kwargs = {}
            get_output_filename_kwargs['subject_no'] = self.per_trial_info[idx]['subject_no']
            get_output_filename_kwargs['trial_no'] = self.per_trial_info[idx]['trial_no']
            get_output_filename_kwargs['trial_name'] = self.per_trial_info[idx]['trial']
            output_filename = self.get_output_filename(**get_output_filename_kwargs)
            npy_folder = os.path.join(self.config['output_root_directory'], self.config['npy_folder'], output_filename)
            ensure_dir(npy_folder)

            self.per_trial_info[idx]['processing_record']['trial'] = output_filename

            self.per_trial_info[idx]['video_npy_path'] = os.path.join(npy_folder, "video.npy")

            # Load the continuous labels
            if hasattr(self, 'extract_continuous_label'):
                self.extract_continuous_label_fn(idx, npy_folder)

            # Load the continuous labels
            if hasattr(self, 'extract_class_label'):
                self.extract_class_label_fn(self.per_trial_info[idx])

            # EEG processing
            if hasattr(self, 'extract_eeg'):
                self.extract_eeg_fn(idx, output_filename, npy_folder)

        path = os.path.join(self.config['output_root_directory'], 'processing_records.pkl')
        ensure_dir(path)
        save_to_pickle(path, self.per_trial_info)

        self.generate_dataset_info()

    def extract_eeg_fn(self, idx, output_filename, npy_folder):

        if self.per_trial_info[idx]['has_eeg']:
            not_done = 0
            for feature in self.config['eeg_config']['features']:
                filename = os.path.join(npy_folder, feature + ".npy")
                if not os.path.isfile(filename):
                    not_done = 1

            if "eeg_processed_path" in self.per_trial_info[idx]:
                output_path = os.path.join(self.config['output_root_directory'],
                                           self.per_trial_info[idx]['eeg_processed_path'][-2:])
            else:
                input_path = os.path.join(self.config['root_directory'], self.config['raw_data_folder'],
                                          *self.per_trial_info[idx]['eeg_path'][-2:])

                output_path = os.path.join(self.config['output_root_directory'], self.config['eeg_folder'],
                                           output_filename)
                if self.per_trial_info[idx]['has_eeg']:   # self.per_trial_info['has_eeg']
                    ensure_dir(output_path)

                    if not_done:
                        from base.eeg import GenericEegController
                        eeg_handler = GenericEegController(input_path, config=self.config['eeg_config'])

            self.per_trial_info[idx]['processing_record']['eeg_processed_path'] = output_path.split(os.sep)
            self.per_trial_info[idx]['eeg_processed_path'] = output_path.split(os.sep)

            if self.config['save_npy'] and self.per_trial_info[idx]['has_eeg']:
            #if self.config['save_npy'] and self.per_trial_info['has_eeg']:
                for feature in self.config['eeg_config']['features']:

                    filename = os.path.join(npy_folder, feature + ".npy")
                    self.per_trial_info[idx]['eeg_' + feature + '_npy_path'] = filename
                    if not os.path.isfile(filename):
                        # Save EEG features npy
                        feature_np = eeg_handler.extracted_data[feature]
                        np.save(filename, feature_np)

    def generate_dataset_info(self):

        for idx, record in self.per_trial_info.items():
            self.dataset_info['trial'].append(record['processing_record']['trial'])
            self.dataset_info['trial_no'].append(record['trial_no'])
            self.dataset_info['subject_no'].append(record['subject_no'])
            self.dataset_info['length'].append(len(self.per_trial_info[idx]['continuous_label']))
            self.dataset_info['partition'].append(record['partition'])

        self.dataset_info['multiplier'] = self.config['multiplier']
        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], 'dataset_info.pkl')
        save_to_pickle(path, self.dataset_info)

    def extract_continuous_label_fn(self, idx, npy_folder):

        raw_continuous_label = self.load_continuous_label(self.per_trial_info[idx]['continuous_label_path'])

        self.per_trial_info[idx]['continuous_label'] = raw_continuous_label[self.per_trial_info[idx]['annotated_index']]

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "continuous_label.npy")
            if not os.path.isfile(filename):
                ensure_dir(filename)
                np.save(filename, self.per_trial_info[idx]['continuous_label'])

    def extract_class_label_fn(self, record):
        pass

    def load_continuous_label(self, path, **kwargs):
        raise NotImplementedError

    def process_continuous_label(self, continuous_label):
        return list(range(len(continuous_label)))

    def generate_iterator(self):
        return NotImplementedError

    def generate_per_trial_info_dict(self):
        raise NotImplementedError

    def get_video_trim_range(self):
        trim_range = []
        return trim_range

    def get_annotated_index(self, annotated_index):
        return annotated_index

    @staticmethod
    def get_output_filename(**kwargs):

        output_filename = "P{}-T{}".format(kwargs['subject_no'], kwargs['trial_no'])
        return output_filename

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "trial": [],
            "subject_no": [],
            "trial_no": [],
            "length": [],
            "partition": [],
        }
        return dataset_info
