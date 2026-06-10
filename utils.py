import os.path
from itertools import compress
from typing import Dict
from typing import Optional
from mne.io import RawArray, concatenate_raws, read_raw_edf
import mne
import numpy as np
# from braindecode.datasets import MOABBDataset
# from braindecode.preprocessing import (
#     Preprocessor,
#     create_windows_from_events,
#     preprocess,
#     scale,
# )

import os
import scipy.io as scio
import torch.nn as nn
# standard package
import numpy as np
import random
random.seed(0)
import copy
import scipy
# DL
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, Optional
from scipy.signal import filtfilt, butter
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


class BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BaseDataModule, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])

    @staticmethod
    def _z_scale(X, X_test):
        for ch_idx in range(X.shape[1]):
            sc = StandardScaler()
            X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
            X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
        return X, X_test

    @staticmethod
    def _make_tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))
    

dataset_path = {'seed4': '/share/data/emotion/SEED_IV/SEED_IV/eeg_feature_smooth', 'seed3': '/home/cjxu/code/data/SEED/ExtractedFeatures'}

'''
Tools
'''
def norminx(data):
    '''
    description: norm in x dimension
    param {type}:
        data: array
    return {type} 
    '''    
    for i in range(data.shape[0]):
        data[i] = normalization(data[i])
    return data

def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T
def normalization(data):
    '''
    description: 
    param {type} 
    return {type} 
    '''
    data_mean = np.mean(data)
    data_std = np.std(data)
    return (data - data_mean) / data_std
    
    # _range = np.max(data) - np.min(data)
    # return (data - np.min(data)) / _range


# package the data and label into one class
class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    # get the size of data
    def __len__(self):
        return len(self.Data)
    # get the data and label
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label

# mmd loss and guassian kernel
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

# new mmd method
min_var_est = 1e-8

def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
    loss = delta.dot(delta)
    return loss
    # delta = f_of_X - f_of_Y
    # loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    # return loss


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4*d*4)
    return loss
    
def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))

def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type} 
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
                    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
                    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
    label_seed3 = [[2,1,0,0,1,2,0,1,2,2,1,0,1,2,0],
                    [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0],
                    [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')

def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''    
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        one_data = np.reshape(np.transpose(data[i], (1,2,0)), (-1,310), order='F')
        one_label = np.full((one_data.shape[0],1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label

def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}: 
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key:value for key, value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label

def sample_by_value(list, value, number):
    '''
    @Description: sample the given list randomly with given value
    @param {type}: 
        list: list
        value: int {0,1,2,3}
        number: number of sampling
    @return: 
        result_index: list
    '''
    result_index = []
    index_for_value = [i for (i,v) in enumerate(list) if v==value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index

'''
For loading data
'''
def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)
    '''    
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats

# load SHU_dataset
# butter worth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=1)
    return y

def load_physionet_data():
    data = []
    label = []
    sub_ids = list(range(1, 110))
    removed_sub = [38, 88, 89, 92, 100, 104]
    for i in range(len(removed_sub)):
        sub_ids.remove(removed_sub[i])

    for sub_num in sub_ids:
        sub_num = str(sub_num).zfill(3)
        sub_data = []
        sub_label = []
        for run_num in [4, 8, 12]:  # imagine opening and closing left or right fist
            run_num = str(run_num).zfill(2)
            data_path = '/home/cjxu/code/data/physionet/files/S' + sub_num + '/S' + sub_num + 'R' + run_num + '.edf'
            raw = mne.io.read_raw_edf(data_path, preload=False)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            eeg = raw.to_data_frame()
            eeg = np.array(eeg)

            for sam_num in range(np.shape(events_from_annot)[0]):
                begin = events_from_annot[sam_num, 0]
                tmp = eeg[begin:begin+640, :]
                if events_from_annot[sam_num, 2] != 1:
                    sub_data.append(tmp)
                    sub_label.append(events_from_annot[sam_num, 2] - 2)

        for run_num in [6, 10, 14]:
            run_num = str(run_num).zfill(2)
            data_path = '/home/cjxu/code/data/physionet/files/S' + sub_num + '/S' + sub_num + 'R' + run_num + '.edf'
            raw = mne.io.read_raw_edf(data_path, preload=False)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            eeg = raw.to_data_frame()
            eeg = np.array(eeg)
            # eeg = eeg[:, [34, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 50, 51, 52, 58]]

            for sam_num in range(np.shape(events_from_annot)[0]):
                if events_from_annot[sam_num, 2] == 3:
                    begin = events_from_annot[sam_num, 0]
                    tmp = eeg[begin:begin + 640, :]
                    sub_data.append(tmp)
                    sub_label.append(events_from_annot[sam_num, 2] - 1)
                elif events_from_annot[sam_num, 2] == 2:
                    begin = events_from_annot[sam_num, 0]
                    tmp = eeg[begin:begin + 640, :]
                    sub_data.append(tmp)
                    sub_label.append(events_from_annot[sam_num, 2] + 1)

        sub_data = np.array(sub_data)
        sub_label = np.array(sub_label)
        sub_data = sub_data[:, :, 1:65]
        sub_data = np.transpose(sub_data, (0, 2, 1))
        sub_label = sub_label.reshape(-1, 1)
        # 标准化
        train_mean = np.mean(sub_data)
        train_std = np.std(sub_data)
        sub_data = (sub_data - train_mean) / train_std
        sub_data = np.expand_dims(sub_data, axis=1)
        print('The subjects id is:', sub_num)
        print('The shape of sub_data is:', sub_data.shape)
        print('The shape of sub_label is:', sub_label.shape)
        data.append(sub_data)
        label.append(sub_label)
    data = np.array(data)
    label = np.array(label)
    print('The shape of data is:', data.shape)
    print('The shape of label is:', label.shape)
    return data, label

def load_shu_data():
    path = '/home/cjxu/code/data/shu_MI_data/mat/'
    data = [([0] * 25) for i in range(2)]
    label = [([0] * 25) for i in range(2)]
    for i in range(25):
        print('Subject: ', i + 1)
        reshape_train_data = None
        reshape_train_label = None
        for j in range(3):
            if i + 1 < 10:
                file_name = path + 'sub-00%d_ses-0%d_task_motorimagery_eeg.mat' % (i + 1, j + 1)
            else:
                file_name = path + 'sub-0%d_ses-0%d_task_motorimagery_eeg.mat' % (i + 1, j + 1)
            raw_data = scipy.io.loadmat(file_name)
            trial_data = raw_data['data']
            trial_label = raw_data['labels']
            # trial_data = np.reshape(np.transpose(trial_data, (1, 0, 2)), (32, -1))
            # trial_data = butter_bandpass_filter(trial_data, lowcut=3, highcut=35, fs=250, order=8)
            # trial_data = np.transpose(np.reshape(trial_data, (32, -1, 1000)), (1, 0, 2))
            trial_label = trial_label.reshape(-1, 1) - 1
            trial_data = np.expand_dims(trial_data, axis=1)
            print('the shape of trial_train_data is: ', trial_data.shape)
            print('the shape of trial_train_label is: ', trial_label.shape)
            if reshape_train_data is not None:
                reshape_train_data = np.vstack((reshape_train_data, trial_data))
                reshape_train_label = np.vstack((reshape_train_label, trial_label))
            else:
                reshape_train_data = trial_data
                reshape_train_label = trial_label
        train_mean = np.mean(reshape_train_data)
        train_std = np.std(reshape_train_data)
        reshape_train_data = (reshape_train_data - train_mean) / train_std
        data[0][i] = reshape_train_data
        label[0][i] = reshape_train_label
        
        reshape_test_data = None
        reshape_test_label = None
        for j in range(3, 5):
            if i + 1 < 10:
                file_name = path + 'sub-00%d_ses-0%d_task_motorimagery_eeg.mat' % (i + 1, j + 1)
            else:
                file_name = path + 'sub-0%d_ses-0%d_task_motorimagery_eeg.mat' % (i + 1, j + 1)
            raw_data = scipy.io.loadmat(file_name)
            trial_data = raw_data['data']
            trial_label = raw_data['labels']
            # trial_data = np.reshape(np.transpose(trial_data, (1, 0, 2)), (32, -1))
            # trial_data = butter_bandpass_filter(trial_data, lowcut=3, highcut=35, fs=250, order=8)
            # trial_data = np.transpose(np.reshape(trial_data, (32, -1, 1000)), (1, 0, 2))
            trial_label = trial_label.reshape(-1, 1) - 1
            trial_data = np.expand_dims(trial_data, axis=1)
            print('the shape of trial_test_data is: ', trial_data.shape)
            print('the shape of trial_test_label is: ', trial_label.shape)
            if reshape_test_data is not None:
                reshape_test_data = np.vstack((reshape_test_data, trial_data))
                reshape_test_label = np.vstack((reshape_test_label, trial_label))
            else:
                reshape_test_data = trial_data
                reshape_test_label = trial_label
        reshape_test_data = (reshape_test_data - train_mean) / train_std
        data[1][i] = reshape_test_data
        label[1][i] = reshape_test_label
    data, label = np.array(data), np.array(label)
    print('data.shape:', data.shape, 'label.shape:', label.shape)
    return data, label

def load_data_2b(data_path):
    data = [([0] * 9) for i in range(2)]
    label = [([0] * 9) for i in range(2)]
    # load train data
    for i in range(1, 10):
        train_data = None
        train_label = None
        for j in range(1, 4):
            total_data = scipy.io.loadmat(data_path + 'B0%d0%dT.mat' % (i, j))
            temp_train_data = total_data['data']
            temp_train_label = total_data['label']
            temp_train_data = np.transpose(temp_train_data, (2, 1, 0))
            temp_train_data = np.expand_dims(temp_train_data, axis=1)
            temp_train_label = temp_train_label - 1
            temp_train_data = temp_train_data[:120]
            temp_train_label = temp_train_label[:120]
            if train_data is not None:
                train_data = np.vstack((train_data, temp_train_data))
                train_label = np.vstack((train_label, temp_train_label))
            else:
                train_data = temp_train_data
                train_label = temp_train_label
        data[0][i-1] = train_data
        label[0][i-1] = train_label
    # load test data
    for i in range(1, 10):
        test_data = None
        test_label = None
        for j in range(4, 6):
            total_data = scipy.io.loadmat(data_path + 'B0%d0%dE.mat' % (i, j))
            temp_test_data = total_data['data']
            temp_test_label = total_data['label']
            temp_test_data = np.transpose(temp_test_data, (2, 1, 0))
            temp_test_data = np.expand_dims(temp_test_data, axis=1)
            temp_test_label = temp_test_label - 1
            temp_test_data = temp_test_data[:120]
            temp_test_label = temp_test_label[:120]
            if test_data is not None:
                test_data = np.vstack((test_data, temp_test_data))
                test_label = np.vstack((test_label, temp_test_label))
            else:
                test_data = temp_test_data
                test_label = temp_test_label
        data[1][i-1] = test_data
        label[1][i-1] = test_label

    return np.array(data), np.array(label)

def load_data_2a(data_path):
    data = [([0] * 9) for i in range(2)]
    label = [([0] * 9) for i in range(2)]
    # load train data
    for i in range(1, 10):
        total_data = scipy.io.loadmat(data_path + 'A0%dT.mat' % i)
        temp_train_data = total_data['data']
        temp_train_label = total_data['label']
        temp_train_data = np.transpose(temp_train_data, (2, 1, 0))
        temp_train_data = np.expand_dims(temp_train_data, axis=1)
        temp_train_label = temp_train_label - 1
        data[0][i-1] = temp_train_data
        label[0][i-1] = temp_train_label
    
    # load test data
    for j in range(1, 10):
        total_data = scipy.io.loadmat(data_path + 'A0%dE.mat' % j)
        temp_test_data = total_data['data']
        temp_test_label = total_data['label']
        temp_test_data = np.transpose(temp_test_data, (2, 1, 0))
        temp_test_data = np.expand_dims(temp_test_data, axis=1)
        temp_test_label = temp_test_label - 1
        data[1][j-1] = temp_test_data
        label[1][j-1] = temp_test_label
    
    return np.array(data), np.array(label)


def load_data(dataset_name):
    '''
    description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/' + str(i+1) + '/' + allmats[i][j]
            one_data, one_label = get_data_label_frommat(mat_path, dataset_name, i)
            data[i][j] = np.array(one_data.copy())
            label[i][j] = np.array(one_label.copy())
    return np.array(data), np.array(label)

# load HGD
# def load_hgd(subject_id: int, preprocessing_dict: Dict = None,
#              verbose: str = "WARNING"):
#     dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[subject_id])

#     if preprocessing_dict.get("remove_artifacts", True):
#         # find samples < 800 uV and save masks for later
#         window_dataset = create_windows_from_events(dataset, preload=False)
#         ds_masks = []
#         for ds in window_dataset.datasets:
#             clean_trial_mask = np.max(
#                 np.abs(ds.windows.load_data()._data), axis=(-2, -1)) < 800 * 1e-6
#             ds_masks.append(clean_trial_mask)

#     channels = [
#         "FC5", "FC1", "FC2", "FC6", "C3", "C4", "CP5", "CP1", "CP2", "CP6", "FC3",
#         "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "FFC5h", "FFC3h",
#         "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h",
#         "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "FFC1h", "FFC2h", "FCC1h", "FCC2h",
#         "CCP1h", "CCP2h", "CPP1h", "CPP2h",
#     ]

#     preprocessors = [
#         Preprocessor("pick_channels", ch_names=channels, verbose=verbose),
#         Preprocessor(scale, factor=1e6, apply_on_array=True),  # from uV to V
#         Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
#     ]

#     l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
#     if l_freq is not None or h_freq is not None:
#         preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
#                                           verbose=verbose))

#     preprocess(dataset, preprocessors)

#     # create windows
#     sfreq = dataset.datasets[0].raw.info["sfreq"]
#     trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
#     trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
#     windows_dataset = create_windows_from_events(
#         dataset, trial_start_offset_samples=trial_start_offset_samples,
#         trial_stop_offset_samples=trial_stop_offset_samples, preload=False
#     )

#     if preprocessing_dict.get("remove_artifacts", True):
#         for (mask, ds) in zip(ds_masks, windows_dataset.datasets):
#             ds.windows = ds.windows[mask]
#             ds.y = list(compress(ds.y, mask))

#     return windows_dataset

# def load_HGD():
#     preprocessing_dict = {'sfreq': 250, 'low_cut': 4, 'high_cut': None, 'start': 0.0, 'stop': 0.0,
#                           'remove_artifacts': True,
#                           'z_scale': True,
#                           'batch_size': 64}
#     data = [([0] * 14) for i in range(2)]
#     label = [([0] * 14) for i in range(2)]
#     for i in range(1, 15):
#         dataset = load_hgd(i, preprocessing_dict)
    
#         # split the data
#         splitted_ds = dataset.split("run")
#         train_dataset, test_dataset = splitted_ds["0train"], splitted_ds["1test"]
#         # load the data
#         X = train_dataset.datasets[0].windows.load_data()._data
#         y = np.array(train_dataset.datasets[0].y)
#         X_test = test_dataset.datasets[0].windows.load_data()._data
#         y_test = np.array(test_dataset.datasets[0].y)
#         # scale data
#         if preprocessing_dict["z_scale"]:
#             X, X_test = BaseDataModule._z_scale(X, X_test)
            
#         X = np.expand_dims(X, axis=1)
#         X_test = np.expand_dims(X_test, axis=1)
#         y = y.reshape(-1, 1)
#         y_test = y_test.reshape(-1, 1)
#         data[0][i-1] = X
#         data[1][i-1] = X_test
#         label[0][i-1] = y
#         label[1][i-1] = y_test
#         print('the shape of X:', X.shape)
#         print('the shape of y:', y.shape)
#         print('the shape of X_test:', X_test.shape)
#         print('the shape of y_test:', y_test.shape)
#     data, label = np.array(data), np.array(label)
#     return data, label

def pick_one_data(dataset_name, session_id=1, cd_count=4, sub_id=0):
    '''
    @Description: pick one data from session 2 (or from other sessions), 
    @param {type}:
        session_id: int
        cd_count: int (to indicate the number of calibration data)
    @return: 
        832 for session 1, 851 for session 0
        cd_data: array (x*310, x is determined by cd_count)
        ud_data: array ((832-x)*310, the rest of that sub data)
        cd_label: array (x*1)
        ud_label: array ((832-x)*1)              
    '''
    path, allmats = get_allmats_name(dataset_name)
    mat_path = path+ "/" + str(session_id+1) + "/" + allmats[session_id][sub_id]
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key:value for key, value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values()) # 24 * 62 * x * 5
    cd_list = []
    ud_list = []
    number_trial, number_label, labels = get_number_of_label_n_trial(dataset_name)
    session_label_one_data = labels[session_id]
    for i in range(number_label):
        # 根据给定的label值从label链表中拿到全部的index后根据数量随机采样
        cd_list.extend(sample_by_value(session_label_one_data, i, int(cd_count/number_label)))
    ud_list.extend([i for i in range(number_trial) if i not in cd_list])
    cd_label_list = copy.deepcopy(cd_list)
    ud_label_list = copy.deepcopy(ud_list)
    for i in range(len(cd_list)):
        cd_list[i] = mat_de_data[cd_list[i]]
        cd_label_list[i] = labels[session_id][cd_label_list[i]]
    for i in range(len(ud_list)):
        ud_list[i] = mat_de_data[ud_list[i]]
        ud_label_list[i] = labels[session_id][ud_label_list[i]]
    
    # reshape
    cd_data, cd_label = reshape_data(cd_list, cd_label_list)
    ud_data, ud_label = reshape_data(ud_list, ud_label_list)
    
    return cd_data, cd_label, ud_data, ud_label


def exponential_running_standardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    """
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized


def EMS(X, factor_new=1e-3, init_block_size=1000):

    for i, trial in enumerate(X):
        X[i, :, :] = exponential_running_standardize(trial.T, factor_new=factor_new,
                                                     init_block_size=init_block_size, eps=1e-4).T
    return X
