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
parser.add_argument('--dataset', default='cornell',help='six new datasets')
parser.add_argument('--hiddim', type=int, default=256, help='hidden dims')
parser.add_argument('--fdrop', type=float, default=0.5, help='drop for pubmed feature')
parser.add_argument('--drop', type=float, default=0.8, help='drop for pubmed layers')
parser.add_argument('--learning_rate', type=float, default=0.03, help='init pubmed learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
# 该参数是用于在验证集上取最优的evals个体求平均
parser.add_argument('--evals', type=int, default=10, help='num of evals')
parser.add_argument('--startLength', type=int, default=4, help='num of startArch')
args = parser.parse_args()


#原来的三个小数据集
#取出对应数据集名字和数据集划分数据
datastr=args.dataset
splitstr=splitstr = '../splits/'+args.dataset+'_split_0.6_0.2_'+str(1)+'.npz'

adj, features, labels, idx_train, idx_val, idx_test = load_new_data(datastr, splitstr)

#新增的三个大数据集，airport/blogcatalog/flickr
# adj, features, labels, idx_train, idx_val, idx_test = load_big_data(args.dataset)


adj_nor = aug_normalized_adjacency(adj)
adj_com = aug_compare_adjacency(adj)
adj_sing = adj_com + sp.eye(adj_com.shape[0])
adj_nor = sparse_mx_to_torch_sparse_tensor(adj_nor).float().cuda()
adj_com = sparse_mx_to_torch_sparse_tensor(adj_com).float().cuda()
adj_sing = sparse_mx_to_torch_sparse_tensor(adj_sing).float().cuda()
features = features.cuda()
labels = labels.cuda()
data = adj_nor, adj_com, adj_sing, features, labels

# adj = aug_normalized_adjacency(adj)
# adj = sparse_mx_to_torch_sparse_tensor(adj).float().cuda()
# features = features.cuda()
# labels = labels.cuda()
# data = adj, features, labels

idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
index = idx_train, idx_val, idx_test


best_arch = [2, 3, 4, 2, 4, 3, 2, 2, 2, 2, 4]
class Model(object):
    """A class representing a model."""

    def __init__(self):
        self.arch = None
        self.val_acc = None
        self.test_acc = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return self.arch


def main(cycles):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    """Algorithm for regularized evolution (i.e. aging evolution)."""
    # 设置一个双端队列方便pop掉最老的个体

    history = []

    # 就是迭代轮数，一轮迭代会插入一个新子代并且删去一个最老的个体
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.


        # Create the child model and store it.
        child = Model()
        #随意设置一个长度

        child.arch = best_arch
        child.val_acc, child.test_acc = train_and_eval_change_new(args, child.arch, data, index)
        history.append(child)
        print(child.arch)
        print("the %d iteration's res{val_acc:%f   test_acc:%f}" % (
        len(history), child.val_acc, child.test_acc))



    return history


iteration=1
for it in range(iteration):
    # store the search history
    print("-----------round begin-----------")
    h = main(20)
    acc=[]
    d={}
    res = 0
    for i in range(len(h)):
        if res < h[i].test_acc:
            idx = i
            res = h[i].test_acc
        acc.append(h[i].test_acc*100)
    accs = acc

    # print('the best test_acc is %f' % (res))
    # print('the best acc in %d interation' % (idx))
    # print(h[idx].arch)

print("-------------------------------------------------")
print(max(accs))
print("the result is %f±%f"%(np.mean(accs),np.var(accs)))
