# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import math

def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer/(in_channel*sum(prime_list)))
    return out_channel_expect


def eval_condition(iepoch, print_result_every_x_epoch):
    if (iepoch + 1) % print_result_every_x_epoch == 0:
        return True
    else:
        return False


def eval_model(model, dataloader):
    predict_list = np.array([])
    label_list = np.array([])
    for sample in dataloader:
        y_predict = model(sample[0])
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate(
            (label_list, sample[1].detach().cpu().numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    return acc


def save_to_log(sentence, Result_log_folder, dataset_name):
    father_path = Result_log_folder + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + '_.txt'
    print(path)
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')


def generate_layer_parameter_list(start, end, paramenter_number_of_layer_list, in_channel=1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print('start = ', start, 'which is larger than end = ', end)
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(
            paramenter_number_of_layer, in_channel, prime_list)

        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list)*out_channel

        layer_parameter_list.append(tuples_in_layer)

    tuples_in_layer_last = []
    first_out_channel = len(prime_list)*get_out_channel_number(
        paramenter_number_of_layer_list[0], input_in_channel, prime_list)
    tuples_in_layer_last.append((in_channel, first_out_channel, start))
    tuples_in_layer_last.append((in_channel, first_out_channel, start+1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


def calculate_mask_index(kernel_length_now, largest_kernel_lenght):
    right_zero_mast_length = math.ceil(
        (largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - \
        kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now


def creat_mask(number_of_input_channel, number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right = calculate_mask_index(
        kernel_length_now, largest_kernel_lenght)
    mask = np.ones(
        (number_of_input_channel, number_of_output_channel, largest_kernel_lenght))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(
            in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(i[1], i[0], i[2], largest_kernel_lenght)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class build_layer_with_layer_parameter(nn.Module):
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(
            torch.from_numpy(os_mask), requires_grad=False)

        self.padding = nn.ConstantPad1d(
            (int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)

        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(
            torch.from_numpy(init_weight), requires_grad=True)
        self.conv1d.bias = nn.Parameter(
            torch.from_numpy(init_bias), requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        # self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result



def get_img_dataloader(args):
    args.domain_num = 1
    # rate = 0.2
    # trdatalist, tedatalist = [], []

    # names = args.img_dataset[args.dataset]
    # args.domain_num = len(names)
    # for i in range(len(names)):
    #     if i in args.test_envs:
    #         tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
    #                                        names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
    #     else:
    #         tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
    #                                 names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
    #         l = len(tmpdatay)
    #         if args.split_style == 'strat':
    #             lslist = np.arange(l)
    #             stsplit = ms.StratifiedShuffleSplit(
    #                 2, test_size=rate, train_size=1-rate, random_state=args.seed)
    #             stsplit.get_n_splits(lslist, tmpdatay)
    #             indextr, indexte = next(stsplit.split(lslist, tmpdatay))
    #         else:
    #             indexall = np.arange(l)
    #             np.random.seed(args.seed)
    #             np.random.shuffle(indexall)
    #             ted = int(l*rate)
    #             indextr, indexte = indexall[:-ted], indexall[-ted:]

    #         trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
    #                                        names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
    #         tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
    #                                        names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=None,
    #     batch_size=args.batch_size,
    #     num_workers=args.N_WORKERS)
    #     for env in trdatalist]

    # eval_loaders = [DataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=args.N_WORKERS,
    #     drop_last=False,
    #     shuffle=False)
    #     for env in trdatalist+tedatalist]
    
    a1 = np.load('cdata/x_1500_10.npy')
    a2 = np.load('cdata/x_2700_25.npy')
    b1 = np.load('cdata/y_1500_10.npy')
    b2 = np.load('cdata/y_2700_25.npy')
    c1 = np.load('cdata/z_1500_10.npy')
    c2 = np.load('cdata/z_2700_25.npy')
    g1 = np.load('cdata/gt_1500_10.npy')
    g2 = np.load('cdata/gt_2700_25.npy')
    # a = np.concatenate([a1, a2], axis=0)
    # b = np.concatenate([b1, b2], axis=0)
    # c = np.concatenate([c1, c2], axis=0)
    # g = np.concatenate([g1, g2], axis=0)
    a = a1
    b = b1
    c = c1
    arrays = [a, b, c]
    X = np.stack(arrays, axis=1)
    Y = g1
    # X_train = np.concatenate([X[:40000], X[50000:90000]], axis=0)
    # X_test = np.concatenate([X[40000:50000], X[90000:]], axis=0)
    # Y_train = np.concatenate([Y[:40000], Y[50000:90000]], axis=0)
    # Y_test = np.concatenate([Y[40000:50000], Y[90000:]], axis=0)
    X_train = X[:40000]
    X_test = X[40000:]
    Y_train = Y[:40000]
    Y_test = Y[40000:]
    
    device = "cuda:0"
    X_train = torch.from_numpy(X_train)
    X_train.requires_grad = False
    X_train = X_train.to(device)
    Y_train = torch.from_numpy(Y_train).to(device)


    X_test = torch.from_numpy(X_test)
    X_test.requires_grad = False
    X_test = X_test.to(device)
    Y_test = torch.from_numpy(Y_test).to(device)

    print(X_train.shape)
 


    Max_kernel_size = 89
    start_kernel_size = 1
    paramenter_number_of_layer_list = [8*128*X_train.shape[1], 5*128*256 + 2*256*128]


# add channel dimension to time series data
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
        X_test = X_test.unsqueeze_(1)

    input_shape = X_train.shape[-1]
    n_class = max(Y_train) + 1
    receptive_field_shape= min(int(X_train.shape[-1]/4), Max_kernel_size)
        
    # generate parameter list
    args.layer_parameter_list = generate_layer_parameter_list(start_kernel_size,
                                                             receptive_field_shape,
                                                             paramenter_number_of_layer_list,
                                                             in_channel = int(X_train.shape[1]))



    batch_size = 16

    train_dataset = list(TensorDataset(X_train, Y_train))
    # train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, batch_size)),2), shuffle=True)
    test_dataset = list(TensorDataset(X_test, Y_test))
    # test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, batch_size)),2), shuffle=False)

    train_loaders = [InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=max(int(min(X_train.shape[0] / 10, batch_size)),2)
        num_workers=4)]

    test_loaders = [DataLoader(
        dataset=train_dataset + test_dataset,
        batch_size=max(int(min(X_train.shape[0] / 10, batch_size)),2),
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)]

    return train_loader, test_loader
