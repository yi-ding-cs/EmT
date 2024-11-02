# This is the processing script of THU-EP dataset
import numpy as np
import os
from base.prepare_data import PrepareData
import h5py


class THU(PrepareData):
    def __init__(self, args):
        super(THU, self).__init__(args)
        self.original_order = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
                               'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'CP1', 'CP2', 'CP5',
                               'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2']
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

    def process_data(self, subject_path):
        """
        This function: 1. Process data into correct format, select broadband data.
        Parameters
        ----------
        subject_path: directory of input data with dimension:
                      (time, channel，label dimension, bands(last one to be broad band))
        Returns
        -------
        data: (trial, time, channel)
        """
        arrays = dict()
        f = h5py.File(subject_path)
        for k, v in f.items():
            arrays[k] = np.array(v)
        data = arrays['data']
        data = np.array(data)
        # select the broad band only
        data = data[:, :, :, 5]  # (time, channel, stimuli)

        return data

    def process_rating(self, rating_path, sub):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
                       3. 12 dimensions of label: ((anger, disgust, fear, sadness, amusement, joy, inspiration,
                          tenderness,arousal, valence, familiarity, and liking), stimuli, subject)
        Parameters
        ----------
        rating_path: directory of label file, dimension:(trial, 12，subject)
        sub: subject ID

        Returns
        -------
        label: (trial,)
        """
        arrays = dict()
        f = h5py.File(rating_path)
        for k, v in f.items():
            arrays[k] = np.array(v)
        label = arrays['ratings']
        label = np.array(label)
        if self.label_type == 'A':
            label = label[8, :, :]
        elif self.label_type == 'V':
            label = label[9, :, :]
        label = label[ :, sub]
        if self.args.num_class == 2:
            label = np.where(label <= 3, 0, 1)
            print('Binary label generated!')
        return label

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (28, 32, 7500) label: (28, 4)
        """

        sub_code = str('sub_' + str(sub + 1) + '.mat')
        subject_path = os.path.join(self.data_path, "EEG data", sub_code)
        rating_path = os.path.join(self.data_path, "Ratings", "ratings.mat")
        data = self.process_data(subject_path)
        data = np.moveaxis(data, [-1, 0], [0, -1])
        data = self.reorder_channel(data=data, graph_type=self.args.graph_type, graph_idx=self.graph_idx)
        data = np.asarray(data)
        data = np.moveaxis(data, -1, 1)
        data = list(data)
        label = self.process_rating(rating_path, sub)

        return data, label

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
            data_, label_ = self.load_data_per_subject(sub)

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
