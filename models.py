'''
Description: 
Author: voicebeer
Date: 2020-09-09 00:06:57
LastEditTime: 2021-03-25 03:27:41
'''
import vit_model
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import Trans
import utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# class CFE(nn.Module):
#     def __init__(self):
#         super(CFE, self).__init__()
#         self.module = nn.Sequential(
#             nn.Linear(310, 256),
#             # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Linear(256, 128),
#             # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Linear(128, 64),
#             # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#         )
    
#     def forward(self, x):
#         x = self.module(x)
#         return x

class pre_trained_MLP(nn.Module):
    def __init__(self):
        super(pre_trained_MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        x = self.module(x)
        return x

# class DSFE(nn.Module):
#     def __init__(self):
#         super(DSFE, self).__init__()
#         self.module = nn.Sequential(
#             nn.Linear(64, 32),
#             # nn.ReLU(inplace=True),
#             nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             # nn.LeakyReLU(negative_slope=0.01, inplace=True),
#         )
    
#     def forward(self, x):
#         x = self.module(x)
#         return x

class MSMDAERNet_tsne(nn.Module):
    def __init__(self, pretrained=False, number_of_source=14, number_of_category=4):
        super(MSMDAERNet_tsne , self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type} 
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_class = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor 
            # to extract the domain specific feature of target data

            DSFE_name = 'self.DSFE' + str(mark)
            data_tgt_DSFE = eval(DSFE_name)(data_tgt_CFE)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE)

            for i in range(number_of_source): 
                class_name = 'self.cls_fc_DSC' + str(i)
                data_tgt_class_i = eval(class_name)(data_tgt_DSFE)
                data_tgt_class.append(data_tgt_class_i)
            # discrepency loss
            for i in range(len(data_tgt_class)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_class[mark], dim=1) - 
                        F.softmax(data_tgt_class[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss
            
        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            feature_DSFE = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                feature_DSFE.append(feature_DSFE_i)
                pred.append(eval(DSC_name)(feature_DSFE_i))
                
            return pred, feature_DSFE

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
# EEGNet:
# class CFE(nn.Module):
#     def __init__(self):
#         super(CFE, self).__init__()
#         self.firstConv = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
#             nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#         self.depthwiseConv = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=(62,1), stride=(1,1), groups=16,bias=False),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ELU(alpha=0.1),
#             nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
#             nn.Dropout(p=0.5)
#         )

#     def forward(self, x):
#         out = self.firstConv(x)
#         out = self.depthwiseConv(out)
#         return out

# def pretrained_CFE(pretrained=False):
#     model = CFE()
#     if pretrained:
#         pass
#     return model


# class DSFE(nn.Module):
#     def __init__(self):
#         super(DSFE, self).__init__()
#         self.separableConv = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
#             nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ELU(alpha=0.1),
#             nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
#             nn.Dropout(p=0.5),
#             nn.Flatten(),
#         )

#     def forward(self, x):
#         x =  self.separableConv(x)
#         return x

# TSception
class CFE(nn.Module):
    def __init__(self, input_size=[1,62,200], sampling_rate=200, num_T=15, num_S=15):
        super(CFE, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                        int(self.pool*0.25))
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        return out


def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model

class DSFE(nn.Module):
    def __init__(self, num_S=15):
        super(DSFE, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_fusion = nn.BatchNorm2d(num_S)
    
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
    
    def forward(self, x):
        out = self.fusion_layer(x)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        return out

class cls(nn.Module):
    def __init__(self, num_S=15, hidden=32, dropout_rate=0, num_classes=3):
        super(cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

# from torchsummary import summary
# model = DSFE()
# # print(model(torch.ones([1, 15, 3, 26])).shape)
# summary(model.cuda(),(15,3,26))


class MSMDAERNet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=14, number_of_category=3):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) + '=cls()')
            # exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_class = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor 
            # to extract the domain specific feature of target data
            DSFE_name = 'self.DSFE' + str(mark)
            data_tgt_DSFE = eval(DSFE_name)(data_tgt_CFE)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE)
            
            # try to narrow the distance between the classifiers
            for i in range(number_of_source):
                class_name = 'self.cls_fc_DSC' + str(i)
                data_tgt_class_i = eval(class_name)(data_tgt_DSFE)
                data_tgt_class.append(data_tgt_class_i)
            # discrepency loss
            for i in range(len(data_tgt_class)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_class[mark], dim=1) - F.softmax(data_tgt_class[i], dim=1)
                    ))

            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            # pred_src->(64,3), label_src->(64,1), label_src.squeeze()->(64)
            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())
            # train_acc = (pred_src.argmax(1) == label_src.squeeze()).sum()/label_src.size(0)
            pred_src = F.softmax(pred_src, dim=1)
            pred_src = pred_src.data.max(1)[1]
            train_acc = pred_src.eq(label_src.data.squeeze()).cpu().sum()/label_src.size(0)
            return cls_loss, mmd_loss, disc_loss, train_acc
            
        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            
            return pred

class MEERNtmp(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(MEERNtmp, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor 
            # to extract the domain specific feature of target data
            for i in range(number_of_source): 
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor 
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) - 
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss, data_src_DSFE
            
        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            
            return pred

class DDC(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DDC, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)
        # data_src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            # data_tgt_feature = self.DSFE(data_tgt_feature)
            # loss = utils.mmd_linear(data_src_feature, data_tgt_feature)
            loss += utils.mmd_rbf_accelerate(data_src_feature, data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss

class DAN(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DAN, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)
        # data_src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            loss += utils.mmd(data_src_feature, data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss

class DAN_tsne(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DAN_tsne, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)
        # data_src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            loss += utils.mmd(data_src_feature, data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss, data_src_feature

class DeepCoraltmp(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DeepCoraltmp, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src = self.sharedNet(data_src)
        # data_Src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt = self.sharedNet(data_tgt)
            loss += utils.CORAL(data_src, data_tgt)

        data_src_cls = self.cls_fc(data_src)
        return data_src_cls, loss, data_src, data_tgt

class DeepCoral(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DeepCoral, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)

        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            loss += utils.CORAL(data_src_feature, data_tgt_feature)
        
        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss

class DANN(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DANN, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        self.cls_fc = nn.Linear(64, number_of_category)
        self.domain_fc = AdversarialNetwork(in_feature=64)
    
    def forward(self, data):
        data = self.sharedNet(data)
        clabel_pred = self.cls_fc(data)
        dlabel_pred = self.domain_fc(AdversarialLayer(high_value=1.0).apply(data))
        return clabel_pred, dlabel_pred

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 32)
        self.ad_layer2 = nn.Linear(32, 32)
        self.ad_layer3 = nn.Linear(32, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x
    
class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 2000.0

    @staticmethod
    def forward(ctx, input):
        iter_num = 0
        iter_num += 1
        output = input * 1.0
        # ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, gradOutput):
        coeff = np.float(2.0 * (high - low) / (1.0 + np.exp(alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput
