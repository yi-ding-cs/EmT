# This is the base script to do processing

from base.preprocessing import bandpower, band_pass_cheby2_sos, get_DE, log_power
import os
import os.path as osp
import pickle
import numpy as np
import h5py


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.ROOT = args.ROOT
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.save_path = osp.join(self.ROOT, 'data_processed')
        self.original_order = []
        self.TS = []
        self.filter_bank = None
        self.filter_allowance = None
        self.sampling_rate = args.sampling_rate

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: trial x chan x time
        label: trial x 1
        """
        pass

    def get_graph_index(self, graph_type):
        """
        This function get the graph index according to the graph_type
        Parameters
        ----------
        graph_type: which type of graph to load

        Returns
        -------
        graph_idx: a list of channel names
        """
        pass

    def reorder_channel(self, data, graph_type, graph_idx):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph_type: which type of graphs is utilized
        graph_idx: index of channel names

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        input_subgraph = False

        for item in graph_idx:
            if isinstance(item, list):
                input_subgraph = True

        idx_new = []
        if not input_subgraph:
            for chan in graph_idx:
                idx_new.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx_new.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph_type), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        data_reordered = []
        for trial in data:
            data_reordered.append(trial[idx_new, :])
        return data_reordered

    def label_processing(self, label):
        """
        This function: project the original label into n classes
        This function is different for different datasets
        Parameters
        ----------
        label: (trial, dim)

        Returns
        -------
        label: (trial,)
        """
        pass

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(self.save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.pkl'
        save_path = osp.join(save_path, name)
        file_dict = {
            'data': data,
            'label': label
        }
        with open(save_path, 'wb') as f:
            pickle.dump(file_dict, f)

    def get_filter_banks(self, data, fs, cut_frequency, allowance):
        """
        This function does band-pass on each trials
        Args:
            data: list of time x chan
            fs: sampling rate
            cut_frequency: list of frequency bands [[1, 3], [4, 8], [8, 12], [12.5, 16], [16.5, 20], [20.5, 28], [30, 45]]
            allowance: list of allowance bands [[0.2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        Returns:
            list of (time x chan x f)

        """
        data_filtered = []
        for trial in data:
            data_filtered_this_trial = []
            for band, allow in zip(cut_frequency, allowance):
                data_filtered_this_trial.append(band_pass_cheby2_sos(
                    data=trial, fs=fs,
                    bandFiltCutF=band,
                    filtAllowance=allow,
                    axis=0
                ))

            if self.args.data_format == 'PSD_DE':
                data_filtered_this_trial.append(trial)  # for AMDET

            data_filtered.append(np.stack(data_filtered_this_trial, axis=-1))
        return data_filtered

    def get_features(self, data, feature_type):
        """
        extract features for the data
        :param data: list of (num_segment, num_sequence, time, chan, f)
        :param feature_type: what kind of feature to extract: 'DE', 'power', 'rpower'
        :return: list of (num_segment, segment_length, chan, f)
        """
        features = []
        for trial in data:
            if feature_type == 'DE':
                results = get_DE(trial, axis=-3)
            if feature_type == 'power':
                results = log_power(trial, axis=-3)
            if feature_type == 'rpower':
                results = log_power(trial, axis=-3, relative=True)
            if feature_type == 'PSD' or feature_type == 'rPSD':
                # trial: num_segment, num_sequence, time, chan
                results = np.empty((trial.shape[0], trial.shape[1], trial.shape[3], len(self.filter_bank)))
                for i, seg in enumerate(trial):
                    for j, seq in enumerate(seg):
                        if feature_type == 'rPSD':
                            results[i, j] = bandpower(
                                data=seq.T, fs=self.sampling_rate, band_sequence=self.filter_bank, relative=True
                            )
                        else:
                            results[i, j] = bandpower(
                                data=seq.T, fs=self.sampling_rate, band_sequence=self.filter_bank, relative=False
                            )

            features.append(results)
        return features

    def split_trial(self, data: list, label: list, segment_length: int = 1,
                    overlap: float = 0, sampling_rate: int = 256, sub_segment=0,
                    sub_overlap=0.0) -> tuple:
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: list of (time, chan) or list of (time, chan, f)
        label: list of label
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate
        sub_segment: how long each sub-segment is (e.g. 1s, 2s,...)
        sub_overlap: overlap rate of sub-segment

        Returns
        -------
        data:list of (num_segment, segment_length, chan) or list of (num_segment, segment_length, chan, f)
        label: list of (num_segment,)
        """
        data_segment = sampling_rate * segment_length
        sub_segment = sampling_rate * sub_segment
        data_split = []
        label_split = []

        for i, trial in enumerate(data):
            trial_split = self.sliding_window(trial, data_segment, overlap)
            label_split.append(np.repeat(label[i], len(trial_split)))
            if sub_segment != 0:
                trial_split_split = []
                for seg in trial_split:
                    trial_split_split.append(self.sliding_window(seg, sub_segment, sub_overlap))
                trial_split = np.stack(trial_split_split)
            data_split.append(trial_split)
        assert len(data_split) == len(label_split)
        return data_split, label_split

    def sliding_window(self, data, window_length, overlap):
        """
        This function split EEG data into shorter segments using sliding windows
        Parameters
        ----------
        data: data, channel
        window_length: how long each window is
        overlap: overlap rate

        Returns
        -------
        data: (num_segment, window_length, channel)
        """
        idx_start = 0
        idx_end = window_length
        step = int(window_length * (1 - overlap))
        data_split = []
        while idx_end < data.shape[0]:
            data_split.append(data[idx_start:idx_end])
            idx_start += step
            idx_end = idx_start + window_length
        return np.stack(data_split)


