# -*- coding:utf-8 -*-

import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from sklearn.decomposition import PCA
import numpy as np
import h5py
import random

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device

def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.


def get_dataset(dataset_name):
    if dataset_name == 'Houston':
        DataPath1 = './data/Houston/Houston_HS.mat'
        DataPath2 = './data/Houston/Houston_MS.mat'
        DataPath3 = './data/Houston/Houston_LiDAR.mat'
        Label_train = './data/Houston/Houston_train.mat'
        Label_test = './data/Houston/Houston_test.mat'
        Data1 = loadmat(DataPath1)['Houston_HS']
        Data2 = loadmat(DataPath2)['Houston_MS']
        Data3 = loadmat(DataPath3)['Houston_LiDAR']
        gt_train = loadmat(Label_train)['Houston_train']
        gt_test = loadmat(Label_test)['Houston_test']

        label_name = ["health grass", "stressed grass", "synthetic grass",
            "trees", "soil", "water", "residential", "commercial",
            "road", "highway", "railway", "parking lot 1",
            "parking lot 2", "tennis court", "running track"]

    elif dataset_name == 'Augsburg_City':
        DataPath1 = './data/Augsburg_City/Augsburg_City_HS.mat'
        DataPath2 = './data/Augsburg_City/Augsburg_City_MS.mat'
        DataPath3 = './data/Augsburg_City/Augsburg_City_SAR.mat'
        Label_train = './data/Augsburg_City/Augsburg_City_train_200.mat'
        Label_test = './data/Augsburg_City/Augsburg_City_test_200.mat'
        Data1 = h5py.File(DataPath1)['Augsburg_City_HS']
        Data2 = loadmat(DataPath2)['Augsburg_City_MS']
        Data3 = loadmat(DataPath3)['Augsburg_City_SAR']
        gt_train = loadmat(Label_train)['Augsburg_City_train']
        gt_test = loadmat(Label_test)['Augsburg_City_test']

        label_name = ["surface water", "street network", "urban fabric",
                      "industrial, commercial, and transport",
                      "mine, dump, and construction sites", "artificial vegetated areas",
                      "arable land", "permanent crops",
                      "pastures", "forests", "shrub", "open spaces with no vegetation",
                      "inland wetlands"]

    elif dataset_name == 'Beijing':
        DataPath1 = './data/Beijing/Beijing_HS.mat'
        DataPath2 = './data/Beijing/Beijing_MS.mat'
        DataPath3 = './data/Beijing/Beijing_SAR.mat'
        Label_train = './data/Beijing/Beijing_train.mat'
        Label_test = './data/Beijing/Beijing_test.mat'
        Data1 = loadmat(DataPath1)['Beijing_HS']
        Data2 = loadmat(DataPath2)['Beijing_MS']
        Data3 = loadmat(DataPath3)['Beijing_SAR']
        gt_train = loadmat(Label_train)['Beijing_train']
        gt_test = loadmat(Label_test)['Beijing_test']

        label_name = ["surface water", "street network", "urban fabric",
                        "industrial, commercial, and transport",
                        "mine, dump, and construction sites", "artificial vegetated areas",
                        "arable land", "permanent crops",
                        "pastures", "forests", "shrub", "open spaces with no vegetation",
                        "inland wetlands"]

    elif dataset_name == 'Wuhan':
        DataPath1 = './data/Wuhan_new/Wuhan_HS_new.mat'
        DataPath2 = './data/Wuhan_new/Wuhan_MS_new.mat'
        DataPath3 = './data/Wuhan_new/Wuhan_SAR_new.mat'
        Label_train = './data/Wuhan_new/Wuhan_train_new.mat'
        Label_test = './data/Wuhan_new/Wuhan_test_new.mat'
        Data1 = loadmat(DataPath1)['Wuhan_HS']
        Data2 = loadmat(DataPath2)['Wuhan_MS']
        Data3 = loadmat(DataPath3)['Wuhan_SAR']
        gt_train = loadmat(Label_train)['Wuhan_train']
        gt_test = loadmat(Label_test)['Wuhan_test']

        label_name = ["surface water", "street network", "urban fabric",
                      "industrial, commercial, and transport",
                      "mine, dump, and construction sites", "artificial vegetated areas",
                      "arable land", "permanent crops",
                      "pastures", "forests", "shrub", "open spaces with no vegetation",
                      "inland wetlands"]

    Data1 = (Data1 - np.min(Data1)) / (np.max(Data1) - np.min(Data1))
    Data2 = (Data2 - np.min(Data2)) / (np.max(Data2) - np.min(Data2))
    Data3 = (Data3 - np.min(Data3)) / (np.max(Data3) - np.min(Data3))

    return Data1, Data2, Data3, gt_train, gt_test, label_name


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    # _, pred = output.topk(maxk, 1, True, True)
    pred = output.unsqueeze(1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

def select_points(mask, num_classes, select_type, ratio=None, rngsd1=None):

    select_size = []
    select_pos = {}

    if select_type == 'normal':

        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask==(i+1))
            select_size.append(each_class.shape[0])
            select_pos[i] = each_class

    elif select_type == 'random':

        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask==(i+1))
            lengthi = each_class.shape[0]
            num = range(1, lengthi)

            random.seed(rngsd1)
            nums = random.sample(num, int(lengthi*ratio))
            select_size.append(len(nums))
            select_pos[i] = each_class[nums, :]

    total_select_pos = select_pos[0]
    for i in range(1, num_classes):
        total_select_pos = np.r_[total_select_pos, select_pos[i]] #(695,2)
    total_select_pos = total_select_pos.astype(int)

    return total_select_pos, select_size

def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #中心区域左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("Patch size: {}".format(patch))
    print("Padded image shape: [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    return mirror_hsi


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def prepare_data(mirror_image, label, band, select_point, patch):
    x_select = np.zeros((select_point.shape[0], patch, patch, band), dtype=np.float32)
    y_select = np.zeros(select_point.shape[0], dtype=np.float32)
    for i in range(select_point.shape[0]):
        x_select[i,:,:,:] = gain_neighborhood_pixel(mirror_image, select_point, i, patch)
        y_select[i] = label[select_point[i][0], select_point[i][1]] - 1
    return x_select, y_select

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA, matrix


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))





