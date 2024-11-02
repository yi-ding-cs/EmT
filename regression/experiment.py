from base.experiment import GenericExperiment
from base.utils import load_pickle, ensure_dir
from base.loss_function import CCCLoss
from base.trainer import Trainer
from base.parameter_control import ResnetParamControl
from model.EmT import EmT
from base.dataset import DataArranger, MyDatasetPreLoad
from base.checkpointer import Checkpointer

import os

import numpy as np


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.task = args.task
        self.case = args.case

        # For tcn and lstm on regression tasks.
        self.bandpower_dim = args.bandpower_dim
        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout
        self.num_eeg_chan = args.num_eeg_chan
        self.num_f = args.num_f

        # For parameter control.
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.backbone_mode = args.backbone_mode

        # For convenience
        if self.case == "loso":
            self.num_folds = 24
            if self.folds_to_run[0] == "all":
                self.folds_to_run = np.arange(24)
        elif self.case == "trs":
            self.num_folds = 10
            if self.folds_to_run[0] == "all":
                self.folds_to_run = np.arange(10)
        else:
            raise ValueError("Unknown case!")

    def prepare(self):
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.get_modality()
        self.continuous_label_dim = self.get_selected_continuous_label_dim()
        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()

    def run(self):
        criterion = CCCLoss()

        for fold in iter(self.folds_to_run):
            save_path = os.path.join(self.save_path,
                                     self.experiment_name + "_" + self.model_name + "_" + self.stamp + "_" + str(
                                         fold) + "_" + self.emotion)
            ensure_dir(save_path)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            model = self.init_model()
            dataloaders = self.init_dataloader(fold)

            trainer_kwards = {'device': self.device, 'emotion': self.emotion, 'model_name': self.model_name, 'model': model, 'save_path': save_path, 'fold': fold,
                              'min_epoch': self.config['min_epoch'], 'max_epoch': self.config['max_epoch'], 'early_stopping': self.config['early_stopping'], 'scheduler': self.scheduler,
                              'learning_rate': self.learning_rate, 'min_learning_rate': self.min_learning_rate, 'patience': self.patience, 'batch_size': self.batch_size,
                              'criterion': criterion, 'factor': self.factor, 'verbose': True, 'milestone': 0, 'metrics': self.config['metrics'],
                              'load_best_at_each_epoch': self.config['load_best_at_each_epoch'], 'save_plot': self.config['save_plot']}

            trainer = Trainer(**trainer_kwards)

            parameter_controller = ResnetParamControl(trainer, gradual_release=self.gradual_release,
                                                release_count=self.release_count, backbone_mode=self.backbone_mode)

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            if not trainer.fit_finished:
                trainer.fit(dataloaders, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            if not trainer.fold_finished and 'test' in dataloaders:
                test_kwargs = {'dataloader_dict': dataloaders, 'epoch': None, 'partition': 'test'}
                trainer.test(checkpoint_controller, predict_only=0, **test_kwargs)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, save_path)

    def init_model(self):
        self.init_randomness()
        if self.model_name == 'EmT':
            model = EmT(
                layers_graph=[1, 2], layers_transformer=2, num_adj=2, num_chan=self.num_eeg_chan,
                num_feature=self.num_f, hidden_graph=32, K=3, num_head=32, dim_head=32,
                dropout=self.cnn1d_dropout, num_class=1, graph2token='Linear', encoder_type='Cheby',
            )
        return model

    def init_data_arranger(self):
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug, self.task, self.case, self.seed)
        return arranger

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        dataset = MyDatasetPreLoad(data, continuous_label_dim, self.modality, self.multiplier,
                          self.feature_dimension, self.window_length,
                          mode, mean_std=None, time_delay=self.time_delay)
        return dataset

    def get_modality(self):
        pass

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        dim = 0
        return dim
