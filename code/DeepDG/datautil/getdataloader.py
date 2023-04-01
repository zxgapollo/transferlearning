# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
from torch.utils.data import TensorDataset


def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist+tedatalist]
    
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

    batchsize = 16

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, batch_size)),2), shuffle=True)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, batch_size)),2), shuffle=False)

    return train_loader, test_loader
