from base.preprocessing import GenericDataPreprocessing
from base.utils import expand_index_by_multiplier, load_pickle, save_to_pickle, get_filename_from_a_folder_given_extension, ensure_dir
from base.label_config import *

import os
import scipy.io as sio

import pandas as pd
import numpy as np

import xml.etree.ElementTree as et


class Preprocessing(GenericDataPreprocessing):
    def __init__(self, config):
        super().__init__(config)

    def generate_iterator(self):
        path = os.path.join(self.config['root_directory'], self.config['raw_data_folder'])
        iterator = [os.path.join(path, file) for file in sorted(os.listdir(path), key=float)]
        return iterator

    def generate_per_trial_info_dict(self):

        per_trial_info_path = os.path.join(self.config['output_root_directory'], "processing_records.pkl")
        if os.path.isfile(per_trial_info_path):
            per_trial_info = load_pickle(per_trial_info_path)
        else:
            per_trial_info = {}
            pointer = 0

            sub_trial_having_continuous_label = self.get_sub_trial_info_for_continuously_labeled()
            all_continuous_labels = self.read_all_continuous_label()

            iterator = self.generate_iterator()

            for idx, file in enumerate(iterator):
                kwargs = {}
                this_trial = {}
                print(file)

                time_stamp_file = get_filename_from_a_folder_given_extension(file, "tsv", "All-Data")[0]
                video_trim_range = self.read_start_end_from_mahnob_tsv(time_stamp_file)
                if video_trim_range is not None:
                    this_trial['video_trim_range'] = self.read_start_end_from_mahnob_tsv(time_stamp_file)
                else:
                    this_trial['discard'] = 1
                    continue

                this_trial['has_continuous_label'] = 0
                session = int(file.split(os.sep)[-1])
                subject_no, trial_no = session // 130 + 1, session % 130

                if subject_no == sub_trial_having_continuous_label[pointer][0] and trial_no == sub_trial_having_continuous_label[pointer][1]:
                    this_trial['has_continuous_label'] = 1

                this_trial['continuous_label'] = None
                this_trial['annotated_index'] = None
                annotated_index = np.arange(this_trial['video_trim_range'][0][1])
                if this_trial['has_continuous_label']:

                    raw_continuous_label = all_continuous_labels[pointer]
                    this_trial['continuous_label'] = raw_continuous_label
                    annotated_index = self.process_continuous_label(raw_continuous_label)
                    this_trial['annotated_index'] = annotated_index
                    pointer += 1

                # this_trial['video_path'] = get_filename_from_a_folder_given_extension(file, "avi")[0].split(os.sep)
                # this_trial['extension'] = "mp4"

                # Some trials has no EEG recordings
                this_trial['has_eeg'] =  1
                eeg_path = get_filename_from_a_folder_given_extension(file, "bdf")
                if len(eeg_path) == 1:
                    this_trial['eeg_path'] = eeg_path[0].split(os.sep)
                else:
                    this_trial['eeg_path'] = None
                    this_trial['has_eeg'] = 0

                this_trial['audio_path'] = ""

                this_trial['subject_no'] = subject_no
                this_trial['trial_no'] = trial_no
                this_trial['trial'] = "P{}-T{}".format(str(subject_no), str(trial_no))

                this_trial['target_fps'] = 64

                kwargs['feature'] = "video"
                kwargs['has_continuous_label'] = this_trial['has_continuous_label']
                this_trial['video_annotated_index'] = self.get_annotated_index(annotated_index, **kwargs)

                this_trial['class_label'] = get_filename_from_a_folder_given_extension(file, "xml")[0]
                per_trial_info[idx] = this_trial

        ensure_dir(per_trial_info_path)
        save_to_pickle(per_trial_info_path, per_trial_info)
        self.per_trial_info = per_trial_info

    def generate_dataset_info(self):

        class_label = {}
        for idx, record in self.per_trial_info.items():
            self.dataset_info['trial'].append(record['processing_record']['trial'])
            self.dataset_info['trial_no'].append(record['trial_no'])
            self.dataset_info['subject_no'].append(record['subject_no'])
            self.dataset_info['has_continuous_label'].append(record['has_continuous_label'])
            self.dataset_info['has_eeg'].append(record['has_eeg'])

            if record['has_continuous_label']:
                self.dataset_info['length'].append(len(record['continuous_label']))
            else:
                self.dataset_info['length'].append(len(record['video_annotated_index']) // 16)

            if self.config['extract_class_label']:
                class_label.update({record['processing_record']['trial']: self.extract_class_label_fn(record)})

        self.dataset_info['multiplier'] = self.config['multiplier']
        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], 'dataset_info.pkl')
        save_to_pickle(path, self.dataset_info)

        if self.config['extract_class_label']:
            path = os.path.join(self.config['output_root_directory'], 'class_label.pkl')
            save_to_pickle(path, class_label)

    def extract_class_label_fn(self, record):
        class_label = {}
        if record['has_eeg']:
            xml_file = et.parse(record['class_label']).getroot()
            felt_emotion = xml_file.find('.').attrib['feltEmo']
            felt_arousal = xml_file.find('.').attrib['feltArsl']
            felt_valence = xml_file.find('.').attrib['feltVlnc']

            arousal = 0 if float(felt_arousal) <= 5 else 1
            valence = 0 if float(felt_valence) <= 5 else 1

            # Arousal_3cls and Valence_3cls are 3-class label determined by felt categorical emotion tags.
            # Arousal and Valence are 2-class label determined by felt arousal and valence intensity.

            class_label = {
                "Arousal": arousal,
                "Valence": valence,
                "Arousal_3cls": arousal_class_to_number[emotion_tag_to_arousal_class[number_to_emotion_tag_dict[felt_emotion]]],
                "Valence_3cls": valence_class_to_number[emotion_tag_to_valence_class[number_to_emotion_tag_dict[felt_emotion]]]
            }

        return class_label

    def extract_continuous_label_fn(self, idx, npy_folder):

        if self.per_trial_info[idx]["has_continuous_label"]:
            raw_continuous_label = self.per_trial_info[idx]['continuous_label']

            if self.config['save_npy']:
                filename = os.path.join(npy_folder, "continuous_label.npy")
                if not os.path.isfile(filename):
                    ensure_dir(filename)
                    np.save(filename, raw_continuous_label)

    def load_continuous_label(self, path, **kwargs):

        cols = [emotion.lower() for emotion in self.config['emotion_list']]

        if os.path.isfile(path):
            continuous_label = pd.read_csv(path, sep=";",
                                           skipinitialspace=True, usecols=cols,
                                           index_col=False).values.squeeze()
        else:
            continuous_label = 0

        return continuous_label

    def get_annotated_index(self, annotated_index, **kwargs):

        feature = kwargs['feature']
        multiplier = self.config['multiplier'][feature]

        if kwargs['has_continuous_label']:
            annotated_index = expand_index_by_multiplier(annotated_index, multiplier)

        # If the trial is not continuously labeled, then the whole facial video is used.
        else:
            pass

        return annotated_index

    def get_sub_trial_info_for_continuously_labeled(self):

        label_file = os.path.join(self.config['root_directory'], "lable_continous_Mahnob.mat")
        mat_content = sio.loadmat(label_file)
        sub_trial_having_continuous_label = mat_content['trials_included']

        return sub_trial_having_continuous_label

    @staticmethod
    def read_start_end_from_mahnob_tsv(tsv_file):
        if os.path.isfile(tsv_file):
            data = pd.read_csv(tsv_file, sep='\t', skiprows=23)
            end = data[data['Event'] == 'MovieEnd'].index[0]
            start_end = [(0, end)]
        else:
            start_end = None
        return start_end

    def read_all_continuous_label(self):
        r"""
        :return: the continuous labels for each trial (dict).
        """

        label_file = os.path.join(self.config['root_directory'], "lable_continous_Mahnob.mat")
        mat_content = sio.loadmat(label_file)
        annotation_cell = np.squeeze(mat_content['labels'])

        label_list = []
        for index in range(len(annotation_cell)):
            label_list.append(annotation_cell[index].T)
        return label_list

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "trial": [],
            "subject_no": [],
            "trial_no": [],
            "length": [],
            "has_continuous_label": [],
            "has_eeg": [],
        }
        return dataset_info


if __name__ == "__main__":
    from configs import config

    pre = Preprocessing(config)
    pre.generate_per_trial_info_dict()
    pre.prepare_data()
