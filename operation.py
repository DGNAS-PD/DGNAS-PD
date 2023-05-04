import random

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#即正则化后的GCN的聚集操作层
class Graph(nn.Module):
    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj
    #即利用正则化矩阵与H(l-1)相乘得到H(l)，T操作只改变H(l)维度
    def forward(self, x):
        x = self.adj.matmul(x)
        return x



#尝试将P操作改成GAT
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            #add_module是Module类的成员函数，输入参数为Module.add_module(name: str, module: Module)。功能为，为Module添加一个子module，对应名字为name
            #add_module()函数也可以在GAT.init(self)以外定义A的子模块

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #输出层的输入张量的shape之所以为(nhid * nheads, nclass)是因为在forward函数中多个注意力机制在同一个节点上得到的多个不同特征被拼接成了一个长的特征

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  #将多个注意力机制在同一个节点上得到的多个不同特征进行拼接形成一个长特征
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) #该行代码即完成原文中的式子（6）
        return F.log_softmax(x, dim=1)  #F.log_softmax在数学上等价于log(softmax(x))，但做这两个单独操作速度较慢，数值上也不稳定。这个函数使用另一种公式来正确计算输出和梯度



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, adj, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj=adj

        self.W = nn.Parameter(torch.empty(size=(out_features, out_features)))#empty（）创建任意数据类型的张量，torch.tensor（）只创建torch.FloatTensor类型的张量
        nn.init.xavier_uniform_(self.W.data, gain=1.414)    #orch.nn.init.xavier_uniform_是一个服从均匀分布的Glorot初始化器,参见：Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))    #遵从原文，a是shape为(2×F',1)的张量
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh) #实现论文中的特征拼接操作 Wh_i||Wh_j ，得到一个shape = (N ， N, 2 * out_features)的新特征矩阵
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #torch.matmul(a_input, self.a)的shape=(N,N,1)，经过squeeze(2)后，shape变为(N,N)

        zero_vec = -9e15*torch.ones_like(e)
        adj=self.adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)   #np.where(condition, x, y),满足条件(condition)，输出x，不满足输出y.
        attention = F.softmax(attention, dim=1) #对每一行内的数据做归一化
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   #当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同
        #h_prime.shape=(N,out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        # repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None)
        # 参数说明：
        # self: 传入的数据为tensor
        # repeats: 复制的份数
        # dim: 要复制的维度，可设定为0/1/2.....
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # repeat方法可以对 Wh 张量中的单维度和非单维度进行复制操作，并且会真正的复制数据保存到内存中
        # repeat(N, 1)表示dim=0维度的数据复制N份，dim=1维度的数据保持不变

        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout, last=False):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout
        self.last = last

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        if not self.last:
            x = F.relu(x)
        return x

#只由P-GCN(聚集)和T(MLP)操作构成
class ModelOp(nn.Module):
    def __init__(self, arch, adj, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp, self).__init__()
        self._ops = nn.ModuleList()
        #默认第一个T预处理也为一个P层
        self._numP = 1
        self._arch = arch
        for element in arch:
            #P为1,T为0
            if element == 1:
                op = Graph(adj)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                print("arch element error")
            self._ops.append(op)
        #将gate按P层的层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5*torch.randn(self._numP), requires_grad=True)
        #一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)
    
    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT  = []
        for i in range(len(self._arch)):
            if i == 0:
                #所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1
                
                res = self._ops[i](res)
                if self._arch[i] == 1:
                    #如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    #如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            #针对第二层起，要考虑其上一层是什么操作
            else:
                #当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    #如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    #如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        #记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                #如果上一层是T
                else:
                    #当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    #如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            #这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits


class ModelOp_GAT(nn.Module):
    def __init__(self, arch, adj, feat_dim, hid_dim, num_classes, fdropout, dropout, alpha):
        super(ModelOp_GAT, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        for element in arch:
            # P为1,T为0
            if element == 1:
                op = GraphAttentionLayer(adj, feat_dim, hid_dim, dropout, alpha)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                print("arch element error")
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = res + sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = res + sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits

class ModelOp_change(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_change, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        self.new_arch=[]
        for element in arch:
            # P为1,T为0
            if element == 1:
                #这里对三种无参传播矩阵进行随机选择
                idx = random.randint(0,2)
                if idx == 0:
                    op = Graph(adj_nor)
                    self.new_arch.append(2)
                elif idx == 1:
                    op = Graph(adj_com)
                    self.new_arch.append(3)
                else:
                    op = Graph(adj_sing)
                    self.new_arch.append(4)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
                self.new_arch.append(0)
            else:
                print("arch element error")

            # if element == 0:
            #     op = MLP(hid_dim, hid_dim, dropout)
            # else:
            #     #代表nor
            #     if element == 2:
            #         op = Graph(adj_nor)
            #     #代表sing
            #     elif element == 3:
            #         op = Graph(adj_sing)
            #     #代表com
            #     else:
            #         op = Graph(adj_com)
            #     self._numP += 1
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits



class ModelOp_change_new(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_change_new, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        # self._arch = arch
        self._arch = []
        for i in arch:
            if i == 0:
                self._arch.append(0)
            else:
                self._arch.append(1)
        for element in arch:
            # P为1,T为0
            idx = element
            if element != 0:
                #这里对三种无参传播矩阵进行随机选择
                if idx == 2:
                    op = Graph(adj_nor)
                elif idx == 3:
                    op = Graph(adj_com)
                else:
                    op = Graph(adj_sing)
                self._numP += 1
            else:
                op = MLP(hid_dim, hid_dim, dropout)

            # if element == 0:
            #     op = MLP(hid_dim, hid_dim, dropout)
            # else:
            #     #代表nor
            #     if element == 2:
            #         op = Graph(adj_nor)
            #     #代表sing
            #     elif element == 3:
            #         op = Graph(adj_sing)
            #     #代表com
            #     else:
            #         op = Graph(adj_com)
            #     self._numP += 1
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits