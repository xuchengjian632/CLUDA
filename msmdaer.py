'''
Description: 
Author: voicebeer
Date: 2020-09-14 01:01:51
LastEditTime: 2021-12-28 01:46:52
'''
# standard
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import random
import time
import math
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import utils
import models
import os
# random seed
from utils import EarlyStopping

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MSMDAER():
    def __init__(self, model=models.MSMDAERNet(), source_loaders=0, target_loader=0, batch_size=64, iteration=200, lr=0.001, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval

    def __getModel__(self):
        return self.model

    def train(self):
        source_iters = []
        for k in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[k]))
        target_iter = iter(self.target_loader)
        correct = 0
        # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
        LEARNING_RATE = self.lr
        # if (i - 1) % 100 == 0:
        #     print("Learning rate: ", LEARNING_RATE)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=self.momentum)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.iteration, eta_min=1e-3*0.1, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,160,240], gamma=0.1, last_epoch=-1)
        # 构建 SummaryWriter
        # writer = SummaryWriter("./runs/test/temp_0.1gamma_1gamma")
        for i in range(1, self.iteration + 1):
            self.model.train()
            try:
                target_data, _ = next(target_iter)
            except Exception as err:
                target_iter = iter(self.target_loader)
                target_data, _ = next(target_iter)
            target_data = target_data.to(device)
            train_actul_num = 0
            source_data = []
            source_label = []
            for j in range(len(source_iters)):
                try:
                    source_data_j, source_label_j = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data_j, source_label_j = next(source_iters[j])
                source_data_j, source_label_j = source_data_j.to(device), source_label_j.to(device)
                source_data.append(source_data_j)
                source_label.append(source_label_j)
                train_actul_num += source_label_j.size(0)
            optimizer.zero_grad()
            cls_loss, mmd_loss, disc_loss, train_acc_num = self.model(data_src=source_data, 
            number_of_source=len(source_iters), data_tgt=target_data, label_src=source_label)

            gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
            # LOSS_WEIGHT = 1.0/(1.0+torch.exp(torch.tensor(100.0-epoch)))
            beta = gamma/10
            # loss = cls_loss + gamma * (mmd_loss + disc_loss)
            # loss = cls_loss + mmd_loss + 0.1 * disc_loss
            loss = cls_loss + gamma * mmd_loss + beta * disc_loss
            # loss = cls_loss + gamma * (mmd_loss)
            # writer.add_scalar('Accuracy/training accuracy', 100. * train_acc_num/train_actul_num, i)
            # writer.add_scalar('Loss/training loss', loss, i)
            # writer.add_scalar('Loss/training cls loss', cls_loss, i)
            # writer.add_scalar('Loss/training mmd loss', mmd_loss, i)
            # writer.add_scalar('Loss/training disc_loss', disc_loss, i)
            # writer.add_scalar('Loss/training gamma', gamma, i)
                
            loss.backward()
            optimizer.step()
            # lr_scheduler.step(iteration)

            # if i % log_interval == 0:
            #     print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'.format(
            #         i, 100.*i/self.iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()
            #     )
            #     )
            if i % (log_interval * 20) == 0:
                t_correct = self.test()
                # writer.add_scalar('Accuracy/test accuracy', 100. * t_correct / len(self.target_loader.dataset), i)
                if t_correct > correct:
                    correct = t_correct
            
            # print('to target max correct: ', correct.item(), "\n")
        # writer.close()
        return 100. * correct / len(self.target_loader.dataset)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data_tgt=data, number_of_source=len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds)/len(preds)    
                test_loss += F.nll_loss(F.log_softmax(pred,dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()
                
                # print("the real target is: {}".format(target.squeeze()))
                # for i in range(len(preds)):
                #     print("source {} classifier predicts: {}".format(i, preds[i].data.max(1)[1]))
                # for i in range(len(preds)):
                #     print("source {} classifier's probability: {}".format(i, preds[i]))
            test_loss /= len(self.target_loader.dataset)
            # writer.add_scalar("Test/Test loss", test_loss, i)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.target_loader.dataset),
                100. * correct / len(self.target_loader.dataset)
            ))
            for n in range(len(corrects)):
                print('Source' + str(n) + 'accnum {}'.format(corrects[n]))
        return correct


def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    train_idxs = list(range(15))
    del train_idxs[subject_id]
    test_idx = subject_id
    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(one_session_label[train_idxs])
    

    # print('Target_subject_id: ', test_idx)
    # print('Source_subject_id: ', train_idxs)
    print("source_data.shape:", source_data.shape, "source_label.shape:", source_label.shape)
    print("target_data.shape:", target_data.shape, "target_label.shape:", target_label.shape)
    print("Target_subject_id:", test_idx)
    del one_session_label
    del one_session_data

    source_loaders = []

    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)

    model = MSMDAER(model=models.MSMDAERNet(pretrained=False, number_of_source=len(source_loaders),
                    number_of_category=category_number),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration = iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    # gpus = [0, 1]
    # model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    # model = nn.DataParallel(model)
    # model = model.to(device)
    # model.to(device)
    # model = nn.DataParallel(model)
    # print(model.__getModel__())
    acc = model.train()
    print('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-MDAER parameters')
    parser.add_argument('--dataset', type=str, default='seed3',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--norm_type', type=str, default='sample',
                        help='the normalization type used for data, "ele", "ems", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type
    # data preparation
    print('Model name: MS-MDAER. Dataset name: ', dataset_name)
    # data, label = utils.load_data(dataset_name)
    # np.save('SEED_raw_data_filter_nor.npy', data)
    # np.save('SEED_raw_label_filter_nor.npy', label)
    # 第一次处理完数据保存在当地，下次可以直接load
    data = np.load('SEED_raw_data_filter_nor.npy')
    label = np.load('SEED_raw_label_filter_nor.npy')
    print("the shape of data:", data.shape)
    print("the label of label:", label.shape)
    data_tmp = copy.deepcopy(data)
    label_tmp = copy.deepcopy(label)
    # print('Normalization type: ', bn)
    # if bn == 'ele':
    #     data_tmp = copy.deepcopy(data)
    #     label_tmp = copy.deepcopy(label)
    #     for i in range(len(data_tmp)):
    #         for j in range(len(data_tmp[0])):
    #             data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    # elif bn == 'sample':
    #     data_tmp = copy.deepcopy(data)
    #     label_tmp = copy.deepcopy(label)
    #     for i in range(len(data_tmp)):
    #         for j in range(len(data_tmp[0])):
    #             data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    # elif bn == 'global':
    #     data_tmp = copy.deepcopy(data)
    #     label_tmp = copy.deepcopy(label)
    #     for i in range(len(data_tmp)):
    #         for j in range(len(data_tmp[0])):
    #             data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    # elif bn == 'ems':
    #     data_tmp = copy.deepcopy(data)
    #     label_tmp = copy.deepcopy(label)
    #     for i in range(len(data_tmp)):
    #         for j in range(len(data_tmp[0])):
    #             data_tmp[i][j] = utils.EMS(data_tmp[i][j], factor_new=1e-3, init_block_size=200)
    # elif bn == 'none':
    #     data_tmp = copy.deepcopy(data)
    #     label_tmp = copy.deepcopy(label)
    # else:
    #     pass
    trial_total, category_number, _ = utils.get_number_of_label_n_trial(dataset_name)
    print("Category_number:", category_number)
    # training settings
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch*3394/batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch*820/batch_size)
    else:
        iteration = 5000
    print('Iteration: {}'.format(iteration))

    # store the results
    csub = []
    csesn = []
    data_tmp = data_tmp.reshape(3,15,3394,1,62,200)
    # cross-validation, LOSO
    # for session_id_main in range(3):
    #     for subject_id_main in range(15):
    #         csub.append(cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
    #                                   batch_size, iteration, lr, momentum, log_interval))
    
    csub.append(cross_subject(data_tmp, label_tmp, 0, 0, category_number,
                                       batch_size, iteration, lr, momentum, log_interval))

    print("Cross-subject: ", csub)
    # print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
    print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
