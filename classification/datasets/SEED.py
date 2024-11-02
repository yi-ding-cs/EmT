# This is the processing script of SEED dataset

import glob
import os
import os.path as osp
import numpy as np
import scipy.io as sio
import pickle
from base.prepare_data import PrepareData


class SEED(PrepareData):
    def __init__(self, args):
        super(SEED, self).__init__(args)
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data_path = args.data_path
        self.loading_key = args.loading_key
        self.original_order = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
                               'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
                               'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
                               'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
                               'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

        self.graph_idx = self.get_graph_index(args.graph_type)

        self.filter_bank = [[1, 3], [4, 8], [8, 12], [12, 16], [16, 20], [20, 28], [30, 45]]

        self.filter_allowance = [[0.2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

    def get_graph_index(self, graph_type):
        """
        This function get the graph index according to the graph_type
        """
        if graph_type == 'BL':
            graph_idx = self.original_order
        else:
            print("Unknown graph type!")
            graph_idx = None
        return graph_idx

    def load_one_session(self, file_to_load, feature_key):
        """
        This function loads one session of the subject
        Parameters
        ----------
        file_to_load: which file to load
        feature_key: which kind of feature to load

        Returns
        -------
        data: list of (time, chan) or list of (segment, chan, f)
        """
        print('Loading file:{}'.format(file_to_load))
        data = sio.loadmat(file_to_load, verify_compressed_data_integrity=False)
        keys = data.keys()
        keys_to_select = [k for k in keys if feature_key in k]
        data_session = []
        for k in keys_to_select:
            one_trial = data[k]
            if one_trial.ndim == 3:
                one_trial = one_trial.transpose(1, 0, 2)   #(chan, segment, frequency) --> (segment, chan, frequency)
            else:
                one_trial = one_trial.transpose(1, 0)
            data_session.append(one_trial)
        if feature_key == 'eeg':
            temp = [item.transpose(1, 0) for item in data_session]
            temp = self.reorder_channel(
                data=temp, graph_type=self.args.graph_type, graph_idx=self.graph_idx
            )
            temp = [item.transpose(1, 0) for item in temp]
            data_session = temp
        return data_session

    def load_data_per_subject(self, sub, keep_dim=False):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load
        keep_dim: True for keeping the session dimension

        Returns
        -------
        data: (session, trial, segment, chan, f) label: (session, trial)
        """
        sub += 1
        label = sio.loadmat(osp.join(self.data_path, 'label.mat'))['label']
        label += 1
        label = np.squeeze(label)
        files_this_subject = []
        for root, dirs, files in os.walk(self.data_path, topdown=False):
            for name in files:
                if sub < 10:
                    sub_code = name[:2]
                else:
                    sub_code = name[:3]
                if '{}_'.format(sub) == sub_code:
                    files_this_subject.append(name)
        files_this_subject = sorted(files_this_subject)

        data_subject = []
        label_subject = []
        for file in files_this_subject:
            sess = self.load_one_session(
                file_to_load=osp.join(self.data_path, file), feature_key=self.loading_key
            )
            if self.args.num_class == 2:
                idx_keep = np.delete(np.arange(label.shape[-1]), np.where(label == 1)[0])
                sess = [sess[idx] for idx in idx_keep]
                label_selected = [label[idx] for idx in idx_keep]
                label_selected = np.where(np.array(label_selected) == 2, 1, 0)
            else:
                label_selected = label
            if keep_dim:
                data_subject.append(sess)
                label_subject.append(label_selected)
            else:
                data_subject.extend(sess)
                label_subject.extend(list(label_selected))
        return data_subject, label_subject

    def create_dataset(self, subject_list, split=False, sub_split=False, feature=False, band_pass_first=True):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        sub_split: (bool) whether to split one segment's data into shorter sub-segment
        feature: (bool) whether to extract features or not

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.pkl'
        """
        for sub in subject_list:
            if len(self.args.session_to_load) == 3:
                # we will use all three sessions
                data_, label_ = self.load_data_per_subject(sub)
            else:
                # use some sessions
                data_all, label_all = self.load_data_per_subject(sub, keep_dim=True)
                data_, label_ = [], []
                for item in self.args.session_to_load:
                    data_.extend(data_all[item-1])
                    label_.extend(label_all[item-1])

            if band_pass_first:
               data_ = self.get_filter_banks(
                   data=data_, fs=self.args.sampling_rate,
                   cut_frequency=self.filter_bank, allowance=self.filter_allowance
               )

            if split:
                if self.args.sub_segment > 0:
                    assert sub_split > 0, "Please set the sub-split as 1 to split the segment into sub-segment"
                data_, label_ = self.split_trial(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate,
                    sub_segment=self.args.sub_segment, sub_overlap=self.args.sub_overlap
                )

            if feature:
                # data_ : list of (segment, sequence, time, chan, f)
                data_ = self.get_features(
                    data=data_, feature_type=self.args.data_format
                )

            print('Data and label for sub{} prepared!'.format(sub))
            self.save(data_, label_, sub)
