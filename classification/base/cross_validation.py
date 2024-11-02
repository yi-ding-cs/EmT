import datetime
import torch
import os
import os.path as osp
from base.train_model import train, test
from base.utils import Averager, get_metrics
import pickle
import numpy as np
from os.path import exists

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        self.text_file = "results_{}.txt".format(args.dataset)
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for {} on {}:\n".format(args.model, args.dataset))
        args_list = list(args.__dict__.keys())
        for i, key in enumerate(args_list):
            file.write("{}){}:{};".format(i, key, args.__dict__[key]))
        file.write('\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data--list of (segment, chan, feature) and label--list of (segment,)
        """
        save_path = osp.join(self.args.ROOT, 'data_processed')
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.pkl'
        path = osp.join(save_path, data_type, sub_code)
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
        data = dataset['data']
        label = dataset['label']
        return data, label

    def prepare_data(self, data_train, label_train, data_test, label_test):
        data_train, label_train = np.concatenate(data_train), np.concatenate(label_train)
        data_test, label_test = np.concatenate(data_test), np.concatenate(label_test)

        data_train, data_test = self.normalize_channel_wise(data_train, data_test)

        data_train, label_train = self.from_numpy_to_tensor(data_train, label_train)
        data_test, label_test = self.from_numpy_to_tensor(data_test, label_test)
        return data_train, label_train, data_test, label_test

    def normalize_channel_wise(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data
        :param test: testing data
        :return: normalized training and testing data
        """
        if train.ndim > 4:
            # MEET has a tensor size of n, seq, chan, x, y
            temp_train = np.reshape(train, (train.shape[0], train.shape[1], train.shape[2], train.shape[3]*train.shape[4]))
            temp_test = np.reshape(test, (test.shape[0], test.shape[1], test.shape[2], test.shape[3] * test.shape[4]))

            for channel in range(temp_train.shape[-1]):
                mean = np.mean(temp_train[:, :, :, channel])
                std = np.std(temp_train[:, :, :, channel])
                if std != 0:
                    temp_train[:, :, :, channel] = (temp_train[:,:,:, channel] - mean) / std
                    temp_test[:, :, :, channel] = (temp_test[:, :, :, channel] - mean) / std
                train = np.reshape(temp_train, train.shape)
                test = np.reshape(temp_test, test.shape)

        elif train.ndim == 4:
            if self.args.data_format == 'PSD_DE':
                for channel in range(train.shape[-2]):
                    mean_PSD = np.mean(train[:, :, channel, :7])
                    std_PSD = np.std(train[:, :, channel, :7])
                    mean_DE = np.mean(train[:, :, channel, 7:])
                    std_DE = np.std(train[:, :, channel, 7:])

                    train[:, :, channel, :7] = (train[:, :, channel, :7] - mean_PSD) / std_PSD
                    test[:, :, channel, :7] = (test[:, :, channel, :7] - mean_PSD) / std_PSD

                    train[:, :, channel, 7:] = (train[:, :, channel, 7:] - mean_DE) / std_DE
                    test[:, :, channel, 7:] = (test[:, :, channel, 7:] - mean_DE) / std_DE
            else:
                for channel in range(train.shape[-2]):
                    mean = np.mean(train[:, :, channel, :])
                    std = np.std(train[:, :, channel, :])
                    train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
                    test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std

        else:
            for channel in range(train.shape[-2]):
                mean = np.mean(train[:, channel, :])
                std = np.std(train[:, channel, :])
                train[:, channel, :] = (train[:, channel, :] - mean) / std
                test[:, channel, :] = (test[:, channel, :] - mean) / std

        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        num_class = int(max(label)) + 1
        index = []
        for i in range(num_class):
            idx_this_class = np.where(label == i)[0]
            if random:
                np.random.shuffle(idx_this_class)
            index.append(idx_this_class)

        idx_train, idx_val = [], []
        for idx in index:
            idx_train.extend(idx[:int(len(idx)*train_rate)])
            idx_val.extend(idx[int(len(idx)*train_rate):])

        # get validation
        val = data[idx_val]
        val_label = label[idx_val]

        train = data[idx_train]
        train_label = label[idx_train]

        return train, train_label, val, val_label

    def from_numpy_to_tensor(self, data, label):
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()
        return data, label

    def leave_sub_out(self, subject=[], shuffle=True, reproduce=False):
        """
        this function achieves leave one subject out cross-validation
        :param subject: the list of subject index
        :param shuffle: whether to shuffle the data before train-val split (not trian-test split!!!)
        :param reproduce: whether to reproduce the previous results
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        ttf = []
        tva = []  # total val accuracy

        for sub in subject:
            va_val = Averager()
            preds, acts = [], []
            data_train, label_train = [], []
            data_test, label_test = self.load_per_subject(sub)
            for sub_ in subject:
                if sub != sub_:
                    data_temp, label_temp = self.load_per_subject(sub_)
                    data_train.extend(data_temp)
                    label_train.extend(label_temp)

            data_train, label_train, data_test, label_test = self.prepare_data(
                data_train=data_train, label_train=label_train, data_test=data_test, label_test=label_test
            )
            print("Training:{}  Test: {}".format(data_train.size(), data_test.size()))
            if reproduce:
                # to reproduce the reported ACC
                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=sub, trial=0)
                acc_val = 0
            else:
                # to train new models
                data_train, label_train, data_val, label_val = self.split_balance_class(
                    data=data_train, label=label_train, train_rate=0.8, random=shuffle
                )

                acc_val = train(
                    args=self.args,
                    data_train=data_train,
                    label_train=label_train,
                    data_val=data_val,
                    label_val=label_val,
                    subject=sub,
                    trial=0)

                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=sub, trial=0)

            va_val.add(acc_val)
            preds.extend(pred)
            acts.extend(act)

            tva.append(va_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc), ttf.append(f1)
            result = '{},{}'.format(acc, f1)
            self.log2txt(result)

        # prepare final report
        results = 'Test mAcc={}({}) mF1={}({}) Val mAcc={}'.format(
            np.mean(tta), np.std(tta), np.mean(ttf), np.std(ttf), np.mean(tva))
        print(results)
        self.log2txt(results)

    def leave_n_sub_out(self, n=8, subject=[], shuffle=False, reproduce=False):
        """
        this function achieves leave n subject out cross-validation
        :param n: number of subjects used for test data
        :param subject: the list of subject index
        :param shuffle: whether to shuffle the data before train-val split (not trian-test split!!!)
        :param reproduce: whether to reproduce the previous results
        """
        # Train and evaluate the model n-subject by n-subject
        tta = []  # total test accuracy
        ttf = []
        tva = []  # total val accuracy
        subjects_list = [subject[i:i + n] for i in range(0, len(subject), n)] # nested list of n subjects each

        if len(subjects_list) > 10:
            combined_list = np.concatenate((subjects_list[-2], subjects_list[-1]))
            # Replace the last two sub-lists with the combined list
            subjects_list = subjects_list[:-2] + [combined_list]

        train_rate = 0.9
        for fold_num, n_subs in enumerate(subjects_list):
            va_val = Averager()
            preds, acts = [], []
            data_train, label_train, data_test, label_test = [], [], [], []

            # select test subjects
            for sub_id in n_subs:
                data_temp, label_temp = self.load_per_subject(sub_id)
                data_test.extend(data_temp)
                label_test.extend(label_temp)

            # select train subjects
            train_idx = [item for item in subject if item not in n_subs]
            for sub_id_ in train_idx:
                data_temp, label_temp = self.load_per_subject(sub_id_)
                data_train.extend(data_temp)
                label_train.extend(label_temp)

            data_train, label_train, data_test, label_test = self.prepare_data(
                data_train=data_train, label_train=label_train, data_test=data_test, label_test=label_test
            )
            print("Training:{}  Test: {}".format(data_train.size(), data_test.size()))
            if reproduce:
                # to reproduce the reported ACC
                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=fold_num, trial=0)
                acc_val = 0
            else:
                # to train new models
                data_train, label_train, data_val, label_val = self.split_balance_class(
                    data=data_train, label=label_train, train_rate=train_rate, random=shuffle
                )

                acc_val = train(
                    args=self.args,
                    data_train=data_train,
                    label_train=label_train,
                    data_val=data_val,
                    label_val=label_val,
                    subject=fold_num,
                    trial=0)

                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                           reproduce=self.args.reproduce,
                                           subject=fold_num, trial=0)

            va_val.add(acc_val)
            preds.extend(pred)
            acts.extend(act)

            tva.append(va_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc), ttf.append(f1)
            result = '{},{}'.format(acc, f1)
            self.log2txt(result)

        # prepare final report
        results = 'Test mAcc={}({}) mF1={}({}) Val mAcc={}'.format(
            np.mean(tta), np.std(tta), np.mean(ttf), np.std(ttf), np.mean(tva))
        print(results)
        self.log2txt(results)

    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()

