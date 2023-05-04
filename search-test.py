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
parser.add_argument('--data', default='wisconsin',help='six new datasets')
parser.add_argument('--hiddim', type=int, default=256, help='hidden dims')
parser.add_argument('--fdrop', type=float, default=0.5, help='drop for pubmed feature')
parser.add_argument('--drop', type=float, default=0.8, help='drop for pubmed layers')
parser.add_argument('--learning_rate', type=float, default=0.03, help='init pubmed learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
# 该参数是用于在验证集上取最优的evals个体求平均
parser.add_argument('--evals', type=int, default=10, help='num of evals')
parser.add_argument('--startLength', type=int, default=4, help='num of startArch')
parser.add_argument('--flag', type=int, default=0, help='determine which kind of dataset')
args = parser.parse_args()

# #取出对应数据集名字和数据集划分数据
datastr=args.data
if args.flag == 0:
    splitstr=splitstr = './splits/'+args.data+'_split_0.6_0.2_'+str(1)+'.npz'
    adj, features, labels, idx_train, idx_val, idx_test = load_new_data(datastr, splitstr)
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_big_data(args.data)
#将P操作增加为三个无参聚合
adj_nor = aug_normalized_adjacency(adj)
adj_com = aug_compare_adjacency(adj)
adj_sing = adj_com + sp.eye(adj_com.shape[0])
adj_nor = sparse_mx_to_torch_sparse_tensor(adj_nor).float().cuda()
adj_com = sparse_mx_to_torch_sparse_tensor(adj_com).float().cuda()
adj_sing = sparse_mx_to_torch_sparse_tensor(adj_sing).float().cuda()
features = features.cuda()
labels = labels.cuda()
data = adj_nor, adj_com, adj_sing, features, labels
# data = adj_nor, features, labels

idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
index = idx_train, idx_val, idx_test


class Model(object):
    """A class representing a model."""

    def __init__(self):
        self.arch = None
        self.val_acc = None
        self.test_acc = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return self.arch


def main(cycles, population_size, sample_size):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    """Algorithm for regularized evolution (i.e. aging evolution)."""
    # 设置一个双端队列方便pop掉最老的个体
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture_new(args.startLength)
        # 返回的是验证集上的平均最优结果和测试集上的最优结果
        model.val_acc, model.test_acc = train_and_eval_change_new(args, model.arch, data, index)
        population.append(model)
        history.append(model)
        print(model.arch)
        print(model.val_acc, model.test_acc)

    # 就是迭代轮数，一轮迭代会插入一个新子代并且删去一个最老的个体
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        # 随机采样作为父本群
        while len(sample) < sample_size:
            candidate = random.choice(list(population))
            sample.append(candidate)

        # 选择父本群中验证集acc最高的个体作为变异
        parent = max(sample, key=lambda i: i.val_acc)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch_multi(parent.arch, np.random.randint(0, 4))
        child.val_acc, child.test_acc = train_and_eval_change_new(args, child.arch, data, index)
        population.append(child)
        history.append(child)
        print(child.arch)
        print("the %d iteration's res{val_acc:%f   test_acc:%f}" % (
        len(history) - len(population), child.val_acc, child.test_acc))

        # Remove the oldest model.
        population.popleft()

    return history





h = main(500, 20, 3)
acc = {}
d = {}
res = 0
for i in range(len(h)):
    hstr = l2s(h[i].arch)
    acc[hstr] = h[i].val_acc * 100
#根据val-acc将最优的十个结构进行选择
accs = dict(sorted(acc.items(), key= lambda x:x[0],reverse=True))
accs = dict_slice(accs,0,10)
#重复进行5轮实验来验证这十个结构哪个为最优
iteration=5
res =np.zeros(10)
for it in range(iteration):
    num = 0
    for sarch in accs.keys():
        arch = s2l(sarch)
        val_acc, test_acc = train_and_eval_change_new(args, arch, data, index)
        print("this arch:")
        print(arch)
        print("the val_acc is %f" %(val_acc))
        res[num] += val_acc
        num += 1
res = np.array([i/iteration for i in res])
idx = np.argmax(res)

#找到最优结构
best_arch = s2l(list((accs.keys()))[idx])
print("最优结构是:")
print(best_arch)


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

