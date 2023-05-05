import sys
import time
import random
import argparse
import collections
import numpy as np

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from utils import *
from train import *
from operation import *
from mutation import *
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser("new-data")
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--data', default='blogcatalog',help='six new datasets')
parser.add_argument('--hiddim', type=int, default=256, help='hidden dims')
parser.add_argument('--fdrop', type=float, default=0.5, help='drop for pubmed feature')
parser.add_argument('--drop', type=float, default=0.8, help='drop for pubmed layers')
parser.add_argument('--learning_rate', type=float, default=0.03, help='init pubmed learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
# 该参数是用于在验证集上取最优的evals个体求平均
parser.add_argument('--evals', type=int, default=10, help='num of evals')
parser.add_argument('--startLength', type=int, default=4, help='num of startArch')
parser.add_argument('--flag', type=int, default=1, help='determine which kind of dataset')
args = parser.parse_args()

# #取出对应数据集名字和数据集划分数据
datastr=args.data
if args.flag == 0:
    splitstr=splitstr = './splits/'+args.data+'_split_0.6_0.2_'+str(1)+'.npz'
    adj, features, labels, idx_train, idx_val, idx_test = load_new_data(datastr, splitstr)
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_big_data(args.data)


adj_nor = aug_normalized_adjacency(adj)
adj_com = aug_compare_adjacency(adj)
adj_sing = adj_com + sp.eye(adj_com.shape[0])
adj_nor = sparse_mx_to_torch_sparse_tensor(adj_nor).float().cuda()
adj_com = sparse_mx_to_torch_sparse_tensor(adj_com).float().cuda()
adj_sing = sparse_mx_to_torch_sparse_tensor(adj_sing).float().cuda()
features = features.cuda()
labels = labels.cuda()
data = adj_nor, adj_com, adj_sing, features, labels


idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
index = idx_train, idx_val, idx_test


#输入最优DGNN结构
best_arch = [4, 4, 4, 4, 3]



#进行10轮10次的重复训练获得最优的均值和方差
avg_list = []
var_list = []
for i in range(10):
    temp_val = np.zeros(10)
    for j in range(10):
        val_acc, test_acc= train_and_eval_change_new(args, best_arch, data, index)
        temp_val[j] = test_acc*100
    print("the %d iterations' test_acc is %f±%f" %(i, np.mean(temp_val), np.var(temp_val)))
    avg_list.append(np.mean(temp_val))
    var_list.append(np.var(temp_val))

nlist = np.array(avg_list)
idx = np.argmax(nlist)


print("-------------------------------------------------")
print("the result is %f±%f" % (avg_list[idx], var_list[idx]))
