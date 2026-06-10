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

import utils
from models import CLUDA
import os

# CLUDA refers to VoiceBeer/MS-MDA: https://github.com/VoiceBeer/MS-MDA

# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

# writer = SummaryWriter()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MSMDA():
    def __init__(self, model=CLUDA(), target_id=0, source_loaders=0, target_loader=0, 
            test_loader=0, batch_size=20, iteration=1000, lr=0.0002, momentum=0.9, log_interval=5):
        self.model = model
        self.model.to(device)
        self.target_id = target_id
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
    
    def __getModel__(self):
        return self.model
    
    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        source_iters = []
        for k in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[k]))
        target_iter = iter(self.target_loader)
        correct = 0
        LEARNING_RATE = self.lr
        # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
        # 构建 SummaryWriter
        writer = SummaryWriter("./runs/physionet/add_contrastive_source/one_branch_multi_source_without_contrastive_loss_between_source_and_target_BS=30_gamma_0.05gamma_1000epoch_T_0.07_0.03_lr_0.0002/target_subject_" + str(self.target_id))
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=self.momentum)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
        
        for itera in range(1, self.iteration+1):
            self.model.train()
            data_src = []
            label_src = []
            # load target data
            try:
                target_data, _ = next(target_iter)
            except Exception as err:
                target_iter = iter(self.target_loader)
                target_data, _ = next(target_iter)
            target_data = target_data.to(device)
            
            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                source_data, source_label = source_data.to(device), source_label.to(device)
                data_src.append(source_data)
                label_src.append(source_label)
            
            cls_loss, contrastive_loss,contrastive_loss_source = self.model(data_tar=target_data, 
                data_src=data_src, label_src=label_src, number_of_source=len(source_iters))
            gamma = 2 / (1 + math.exp(-10 * (itera) / (self.iteration))) - 1
            beta1 = gamma
            beta2 = gamma / 2
            loss = cls_loss + beta1 * contrastive_loss + beta2 * contrastive_loss_source
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Iteration: ', itera, 'Train loss: ', loss.item(), 'cls_loss: ', cls_loss.item(), 'Contrastive loss: ', contrastive_loss.item(), 'Contrastive loss source: ', contrastive_loss_source.item())
            # gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
            # beta = gamma/10
            # loss = cls_loss + gamma * (mmd_loss + l1_loss)
            
            writer.add_scalar('training gamma', gamma, itera)
            writer.add_scalar('Loss/training loss', loss, itera)
            writer.add_scalar('Loss/training cls_loss', cls_loss, itera)
            writer.add_scalar('Loss/training contrastive_loss', contrastive_loss, itera)
            writer.add_scalar('Loss/training contrastive_loss_source', contrastive_loss_source, itera)
            
            # writer.add_scalar('Loss/training acc', 100. * train_pred_num/train_actul_num, i)
            # writer.add_scalar('Loss/training aver acc', 100. * (train_pred_num/8)/(train_actul_num/8), i)
            
            
            # if i % log_interval == 0:
            #     print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'.format(
            #         i, 100.*i/self.iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()
            #     )
            #     )
            
            if itera % self.log_interval == 0:
                t_correct, test_loss = self.test(itera)
                if t_correct > correct:
                    correct = t_correct
                # print('Iteration: ', itera, 'Train loss: ', loss.item())
                writer.add_scalar('Loss/test loss', test_loss, itera)
                print('Iteration: ', itera, 'Train loss: ', loss.item(), 'cls_loss: ', cls_loss.item(), 'Contrastive loss source: ', contrastive_loss_source.item())
                # print('Iteration: ', itera, 'Train loss: ', loss.item(), 'cls_loss: ', cls_loss.item(), 'Contrastive loss: ', contrastive_loss.item(), 'Contrastive loss source: ', contrastive_loss_source.item())
                # print('to target max correct: ', correct.item(), "\n")
        writer.close()
        return 100. * correct / len(self.test_loader.dataset)
    
    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        # corrects = []
        # for j in range(len(self.source_loaders)):
        #     corrects.append(0)
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(device)
                target = target.to(device)
                pred = self.model(data_tar=data, number_of_source=len(self.source_loaders))
                pred = F.softmax(pred, dim=1)
                test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                # for j in range(len(self.source_loaders)):
                #     pred = preds[j].data.max(1)[1]
                #     corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.test_loader)
            # writer.add_scalar("Test/Test loss", test_loss, i)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)
            ))
            # for n in range(len(corrects)):
            #     print('Source' + str(n) + 'accnum {}'.format(corrects[n]))
        return correct, test_loss

def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    # 训练时使用所有subject的训练数据
    # one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    train_idxs = list(range(103))
    del train_idxs[subject_id]
    test_idx = subject_id

    source_data, source_label = copy.deepcopy(data[train_idxs]), copy.deepcopy(label[train_idxs])
    target_data, target_label = copy.deepcopy(data[test_idx]), copy.deepcopy(label[test_idx])
    test_data, test_label = copy.deepcopy(data[test_idx]), copy.deepcopy(label[test_idx])
    # print('Target_subject_id: ', test_idx)
    # print('Source_subject_id: ', train_idxs)
    print("source_data.shape:", source_data.shape, "source_label.shape:", source_label.shape)
    print("target_data.shape:", target_data.shape, "target_label.shape:", target_label.shape)
    print("test_data.shape:", test_data.shape, "test_label.shape:", test_label.shape)
    print("Target_subject_id:", test_idx)

    source_loaders = []

    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=False))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(test_data, test_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=False)
    model = MSMDA(model=CLUDA(number_of_source=len(source_loaders),
                    number_of_category=category_number, T=0.07, T2=0.03),
                    target_id=test_idx,
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    test_loader=test_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    
    # gpus = [0, 1]
    # model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    # model = model.to(device)
    # model.to(device)
    # model = nn.DataParallel(model)
    # print(model.__getModel__())
    acc = model.train()
    print('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    return acc

if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    parser = argparse.ArgumentParser(description='MS-MDAER parameters')
    parser.add_argument('--dataset', type=str, default='Physionet EEG Motor Movement/Imagery Dataset',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--category_number', type=int, default=4,
                        help='size of category_number, integer')
    parser.add_argument('--norm_type', type=str, default='none',
                        help='the normalization type used for data, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type

    # data preparation
    print('Model name: MS-MDAER. Dataset name: ', dataset_name)
    
    data, label = utils.load_physionet_data()
    # for i in range(data.shape[1]):
    #     print('the shape of data[0][i] is: ', data[0][i].shape)
    # for j in range(data.shape[1]):
    #     print('the shape of data[1][j] is: ', data[1][j].shape)
    # The shape of data is: (104, 90, 64, 640)
    # The shape of label is: (104, 90, 1)
    print('the shape of all data:', data.shape)
    print('the shape of all label:', label.shape)
    print('Normalization type: ', bn)
    if bn == 'ele':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    elif bn == 'sample':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    elif bn == 'global':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    elif bn == 'none':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
    elif bn == 'EMS':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.EMS(data_tmp[i][j], factor_new=1e-3, init_block_size=1000)
    else:
        pass

    # two category
    category_number = args.category_number
    # training settings
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = int(90 / batch_size)
    iteration = math.ceil((epoch*90) / batch_size)
    print('Iteration: {}'.format(iteration))
    
    # store the results
    csub = []
    # cross-validation, LOSO
    session_id_main = 0
    for subject_id_main in range(103):
        csub.append(cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
                                    batch_size, iteration, lr, momentum, log_interval))
    print("Cross-subject: ", csub)
    # print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
    print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
    print(time.asctime(time.localtime(time.time())))
