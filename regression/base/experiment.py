from base.utils import load_pickle, save_to_pickle

import pickle
import os
import random

import numpy as np
import torch
from torch import optim


class GenericExperiment(object):
    def __init__(self, args):

        # Basic experiment settings.
        self.experiment_name = args.experiment_name
        self.dataset_path = args.dataset_path
        self.load_path = args.load_path
        self.save_path = args.save_path
        self.stamp = args.stamp
        self.seed = args.seed
        self.resume = args.resume
        self.debug = args.debug
        self.test = args.test
        self.calc_mean_std = args.calc_mean_std

        self.high_performance_cluster = args.high_performance_cluster
        self.gpu = args.gpu
        self.cpu = args.cpu
        self.device = self.init_device()
        # If the code is to run on high-performance computer, which is usually not
        # available to specify gpu index and cpu threads, then set them to none.
        if self.high_performance_cluster:
            self.gpu = None
            self.cpu = None

        self.model_name = args.model_name
        self.cross_validation = args.cross_validation
        self.num_folds = args.num_folds
        self.folds_to_run = args.folds_to_run
        if not self.cross_validation:
            self.folds_to_run = [0]

        self.scheduler = args.scheduler
        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.factor = args.factor
        self.patience = args.patience
        self.modality = args.modality
        self.calc_mean_std = args.calc_mean_std

        self.window_length = args.window_length
        self.hop_length = args.hop_length
        self.batch_size = args.batch_size

        self.time_delay = None
        self.dataset_info = None
        self.mean_std_dict = None
        self.data_arranger = None

        self.feature_dimension = None
        self.multiplier = None
        self.continuous_label_dim = None

        self.config = None

        if 'emotion' in args:
            self.emotion = args.emotion

    def prepare(self):

        self.config = self.get_config()
        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.continuous_label_dim = self.get_selected_continuous_label_dim()

        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.init_randomness()
        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()
        self.mean_std_dict = load_pickle(os.path.join(self.dataset_path, "mean_std_info.pkl"))

    def get_config(self):
        raise NotImplementedError

    def get_selected_continuous_label_dim(self):
        dim = [0 if self.emotion == "arousal" else 1]
        return dim

    def run(self):
        raise NotImplementedError

    def init_dataloader(self, fold):
        self.init_randomness()
        data_list = self.data_arranger.generate_partitioned_trial_list(window_length=self.window_length,
                                                                       hop_length=self.hop_length, fold=fold,
                                                                       windowing=True)

        datasets, dataloaders = {}, {}
        for mode, data in data_list.items():
            if len(data):

                # Cross-platform deterministic shuffling for the training set.
                if mode == "train":
                    random.shuffle(data_list[mode])
                datasets[mode] = self.init_dataset(data_list[mode], self.continuous_label_dim, mode, fold)

                dataloaders[mode] = torch.utils.data.DataLoader(
                    dataset=datasets[mode], batch_size=self.batch_size, shuffle=False)

        return dataloaders

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        raise NotImplementedError

    def init_model(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_time_delay(config):
        time_delay = config['time_delay']
        return time_delay

    @staticmethod
    def get_feature_dimension(config):
        feature_dimension = config['feature_dimension']
        return feature_dimension

    @staticmethod
    def get_multiplier(config):
        multiplier = config['multiplier']
        return multiplier

    def get_modality(self):
        raise NotImplementedError

    def init_data_arranger(self):
        raise NotImplementedError

    def get_mean_std_dict_path(self):
        path = os.path.join(self.dataset_path, "mean_std_info.pkl")
        return path

    def calc_mean_std_fn(self):
        path = self.get_mean_std_dict_path()

        mean_std_dict = {}
        for fold in range(self.num_folds):
            data_list = self.data_arranger.generate_partitioned_trial_list(window_length=self.window_length,
                                                                           hop_length=self.hop_length, fold=fold,
                                                                           windowing=False)
            mean_std_dict[fold] = self.data_arranger.calculate_mean_std(data_list)

        save_to_pickle(path, mean_std_dict, replace=True)

    def init_randomness(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.high_performance_cluster:
            torch.cuda.set_device(self.gpu)

        torch.set_num_threads(self.cpu)

        return device


    def init_config(self):
        raise NotImplementedError