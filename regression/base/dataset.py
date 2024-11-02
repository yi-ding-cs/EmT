import torch
import os
from operator import itemgetter

import numpy as np
import random
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset

from base.utils import roll_list


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, Imgs):
        L, C, H, W = Imgs.shape

        tensor = []
        for k in range(L):
            img = self.normalize(Imgs[k, :, :, :])
            tensor.append(img)

        return torch.stack(tensor, dim=0)


class MyDataset(Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=None,
                 time_delay=0, feature_extraction=0):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

    def get_feature_transform(self, feature):
        if "video" in feature:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        return transform

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, trial, length, index = self.data_list[index]

        examples = {}

        for feature in self.modality:
            examples[feature] = self.get_example(path, length, index, feature)

        if len(index) < self.window_length:
            index = np.arange(self.window_length)

        return examples, trial, length, index

    def get_example(self, path, length, index, feature):

        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        # Probably, a trial may be shorter than the window, so the zero padding is employed.
        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        # Sometimes we may want to shift the label, so that
        # the ith label point  corresponds to the (i - time_delay)-th data point.
        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")

        # For the test set, labels of zeros are generated as dummies.
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class MyDatasetPreLoad(Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode, mean_std=None,
                 time_delay=0, feature_extraction=0):
        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

        self.examples_list, self.trial_list, self.length_list, self.index_list = self.pre_load()

    def get_feature_transform(self, feature):
        if "video" in feature:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        return transform

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.examples_list[index], self.trial_list[index], self.length_list[index], self.index_list[index]

    def pre_load(self):
        examples_list, trial_list, length_list, index_list = [], [], [], []
        for index in range(len(self.data_list)):
            path, trial, length, index = self.data_list[index]
            examples = {}
            for feature in self.modality:
                examples[feature] = self.get_example(path, length, index, feature)

            if len(index) < self.window_length:
                index = np.arange(self.window_length)

            examples_list.append(examples)
            trial_list.append(trial)
            length_list.append(length)
            index_list.append(index)

        return examples_list, trial_list, length_list, index_list

    def get_example(self, path, length, index, feature):

        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        # Probably, a trial may be shorter than the window, so the zero padding is employed.
        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        # Sometimes we may want to shift the label, so that
        # the ith label point  corresponds to the (i - time_delay)-th data point.
        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")

        # For the test set, labels of zeros are generated as dummies.
        #data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename)
            else:
                data = np.load(filename)[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class DataArranger(object):
    def __init__(self, dataset_info, dataset_path, debug, task, case, seed):
        self.task = task
        self.case = case
        self.seed = seed
        self.dataset_info = dataset_info
        self.debug = debug
        self.trial_list = self.generate_raw_trial_list(dataset_path)
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()

    def generate_partitioned_trial_list(self, window_length, hop_length, fold, windowing=True):

        partition_range = list(roll_list(self.partition_range, fold))
        partitioned_trial = {}

        trial_list = self.trial_list

        for partition, num_fold in self.fold_to_partition.items():
            partitioned_trial[partition] = []
            if partition == "validate":
                partitioned_trial['validate'] = validate_trial

            for i in range(num_fold):
                index = partition_range.pop(0)
                trial_of_this_fold = list(itemgetter(*index)(trial_list))

                if len(index) == 1:
                    trial_of_this_fold = [trial_of_this_fold]

                for path, trial, length in trial_of_this_fold:
                    partitioned_trial[partition].append([path, trial, length])

            # Split the total N-1 folds into training and validation sets (80% to 20%).

            if partition == "train":
                random.Random(self.seed).shuffle(partitioned_trial['train'])

                count = len(partitioned_trial['train'])

                num_train_trials = int(count * 0.8)

                train_idx = np.arange(num_train_trials)
                validate_idx = np.arange(num_train_trials, count)

                train_trial = list(itemgetter(*train_idx)(partitioned_trial["train"]))
                validate_trial = list(itemgetter(*validate_idx)(partitioned_trial["train"]))

                partitioned_trial['train'] = train_trial

        windowed_partitioned_trial = {key: [] for key in partitioned_trial}
        for partition, trials in partitioned_trial.items():
            for path, trial, length in trials:

                if windowing:
                    windowed_indices = self.windowing(np.arange(length), window_length=window_length,
                                                    hop_length=hop_length)
                else:
                    windowed_indices = self.windowing(np.arange(length), window_length=length,
                                                      hop_length=hop_length)

                for index in windowed_indices:
                    windowed_partitioned_trial[partition].append([path, trial, length, index])

        return windowed_partitioned_trial

    def generate_raw_trial_list(self, dataset_path):
        trial_path = os.path.join(dataset_path, self.dataset_info['data_folder'])
        train_list = []

        for trial in self.generate_iterator():
            idx = self.dataset_info['trial'].index(trial)
            trial = self.dataset_info['trial'][idx]
            path = os.path.join(trial_path, trial)
            length = self.dataset_info['length'][idx]
            train_list.append([path, trial, length])

        return train_list

    def generate_iterator(self):
        iterator = []

        for idx, trial in enumerate(self.dataset_info['trial']):
            if self.task == "reg":
                if self.dataset_info['has_continuous_label'][idx]:
                    iterator.append(trial)
            elif self.task == "cls":
                if self.dataset_info['has_eeg'][idx]:
                    iterator.append(trial)
        return iterator

    def partition_range_fn(self):

        if self.task == "reg":
            if self.case == "trs":
                partition_range = [np.arange(a, a+24) for a in np.arange(0, 239, 24)]

                partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
                partition_range[-2] = np.delete(partition_range[-2], [-1])
                partition_range[-1] = np.delete(partition_range[-1], [-1])

            elif self.case == "loso":
                partition_range = [np.arange(0, 19), np.arange(19, 24), np.arange(24, 37),
                                   np.arange(37, 46), np.arange(46, 59), np.arange(59, 68),
                                   np.arange(68, 81), np.arange(81, 94), np.arange(94, 99),
                                   np.arange(99, 105), np.arange(105, 118), np.arange(118, 120),
                                   np.arange(120, 121), np.arange(121, 132), np.arange(132, 149),
                                   np.arange(149, 159), np.arange(159, 173), np.arange(173, 188),
                                   np.arange(188, 200), np.arange(200, 216), np.arange(216, 222),
                                   np.arange(222, 226), np.arange(226, 236), np.arange(236, 239)]

        elif self.task == "cls":
            if self.case == "trs":
                partition_range = [np.arange(a, a+53) for a in np.arange(0, 527, 53)]
                for i in range(3):
                    partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
                    partition_range[-2] = np.delete(partition_range[-2], [-1])
            elif self.case == "loso":
                partition_range = [np.arange(0, 20), np.arange(20, 40), np.arange(40, 57),
                                   np.arange(57, 77), np.arange(77, 97), np.arange(97, 117),
                                   np.arange(117, 137), np.arange(137, 157), np.arange(157, 171),
                                   np.arange(171, 191), np.arange(191, 211), np.arange(211, 231),
                                   np.arange(231, 251), np.arange(251, 267), np.arange(267, 287),
                                   np.arange(287, 307), np.arange(307, 327), np.arange(327, 347),
                                   np.arange(347, 367), np.arange(367, 387), np.arange(387, 407),
                                   np.arange(407, 427), np.arange(427, 447), np.arange(447, 467),
                                   np.arange(467, 487), np.arange(487, 507), np.arange(507, 527)]
        else:
            raise ValueError("Unknown task!")

        if self.debug == 1:
            if self.case == "loso":
                num_folds = 24
            elif self.case == "trs":
                num_folds = 10
            else:
                raise ValueError("Unknown case!")
            partition_range = [np.arange(a, a + 1) for a in range(num_folds)]

        return partition_range

    def assign_fold_to_partition(self):
        if self.task == "reg":
            if self.case == "trs":
                fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
            elif self.case == "loso":
                fold_to_partition = {'train': 23, 'validate': 0, 'test': 1}
            else:
                raise ValueError("Unknown case!!")
        elif self.task == "cls":
            if self.case == "trs":
                fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
            elif self.case == "loso":
                fold_to_partition = {'train': 26, 'validate': 0, 'test': 1}
            else:
                raise ValueError("Unknown case!!")
        else:
            raise ValueError("Unknown task!")

        return fold_to_partition

    @staticmethod
    def get_feature_list():
        feature_list = ['eeg_bandpower', 'eeg_raw']
        return feature_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        length = len(x)

        if length >= window_length:
            steps = (length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < length - 1:
                sampled_x.append(x[-window_length:])
        else:
            sampled_x = [x]

        return sampled_x