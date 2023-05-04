import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from utils import *
from operation import *

def train_and_eval(args, arch, data, index):
    adj, features, labels = data
    record = []
    test_record = []
    #传入[序列，正则化矩阵，特征数，隐藏层维度，分类数，特征dropout，dropout]
    model = ModelOp(arch, adj, features.shape[1], args.hiddim, labels.max().item() + 1, args.fdrop, args.drop)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #针对当前的model进行epoch轮的训练
    for epoch in range(args.epochs):
        _train_and_eval(model, features, labels, criterion, optimizer, index, record, test_record)
    record.sort()
    record.reverse()
    return sum(record[:args.evals])/args.evals, max(test_record)

def _train_and_eval(model, features, labels, criterion, optimizer, index, record, test_record):
    #告诉模型现在在训练
    model.train()
    optimizer.zero_grad()
    #调用forward进行前向传播
    logits = model(features)
    idx_train, idx_val, idx_test = index
    loss_train = criterion(logits[idx_train], labels[idx_train])
    acc_train = accuracy(logits[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    model.eval()
    logits = model(features)
    acc_val = accuracy(logits[idx_val], labels[idx_val])
    acc_test = accuracy(logits[idx_test], labels[idx_test])

    #logging.info('loss_tra/val acc_ %f %f %f %f', loss_train, loss_val, acc_train, acc_val)
    record.append(acc_val.item())
    test_record.append(acc_test.item())

def train_and_eval_GAT(args, arch, data, index):
    adj, features, labels = data
    record = []
    test_record = []
    #传入[序列，正则化矩阵，特征数，隐藏层维度，分类数，特征dropout，dropout,alpha]
    model = ModelOp_GAT(arch, adj, features.shape[1], args.hiddim, labels.max().item() + 1, args.fdrop, args.drop, args.alpha)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #针对当前的model进行epoch轮的训练
    for epoch in range(args.epochs):
        _train_and_eval(model, features, labels, criterion, optimizer, index, record, test_record)
    record.sort()
    record.reverse()
    return sum(record[:args.evals])/args.evals, max(test_record)

def train_and_eval_change(args, arch, data, index):
    adj_nor, adj_com, adj_sing, features, labels = data
    record = []
    test_record = []
    #传入[序列，正则化矩阵，特征数，隐藏层维度，分类数，特征dropout，dropout,alpha]
    model = ModelOp_change(arch, adj_nor, adj_com, adj_sing, features.shape[1], args.hiddim, labels.max().item() + 1, args.fdrop, args.drop)
    model = model.cuda()
    new_arch = model.new_arch
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #针对当前的model进行epoch轮的训练
    for epoch in range(args.epochs):
        _train_and_eval(model, features, labels, criterion, optimizer, index, record, test_record)
    record.sort()
    record.reverse()
    return sum(record[:args.evals])/args.evals, max(test_record), new_arch


def train_and_eval_change_new(args, arch, data, index):
    adj_nor, adj_com, adj_sing, features, labels = data
    record = []
    test_record = []
    #传入[序列，正则化矩阵，特征数，隐藏层维度，分类数，特征dropout，dropout,alpha]
    model = ModelOp_change_new(arch, adj_nor, adj_com, adj_sing, features.shape[1], args.hiddim, labels.max().item() + 1, args.fdrop, args.drop)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #针对当前的model进行epoch轮的训练
    for epoch in range(args.epochs):
        _train_and_eval(model, features, labels, criterion, optimizer, index, record, test_record)
    record.sort()
    record.reverse()
    return sum(record[:args.evals])/args.evals, max(test_record)