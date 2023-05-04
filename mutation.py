import copy
import random
import numpy as np

def random_architecture(startLength):
    """Returns a random architecture (bit-string) represented as an int."""
    return list(np.random.randint(0, 2, startLength))


def random_architecture_new(startLength):
    """Returns a random architecture (bit-string) represented as an int."""
    a = []
    p = 0.5
    for i in range(startLength):
        if random.random() < p:
            a.append(0)
        else:
            new_value = random.randint(2,4)
            a.append(new_value)
    return a

def mutate_arch(parent_arch, mutate_type):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        child_arch[position] ^= 1
    #随机插入一个P操作
    elif mutate_type == 1:
        position = random.randint(0, length)
        child_arch.insert(position, 1)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length)
        child_arch.insert(position, 0)
    else:
        print('mutate type error')
    return child_arch

def mutate_arch_new(parent_arch, mutate_type):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(0,2)
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        child_arch[position] ^= 1
    #随机插入一个P操作
    elif mutate_type == 1:
        position = random.randint(0, length)
        child_arch.insert(position, 1)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length)
        child_arch.insert(position, 0)
    return child_arch



def mutate_arch_multi(parent_arch, mutate_type):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(0,2)
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        if child_arch[position] == 0:
            new_value = random.randint(2,4)
            child_arch[position] = new_value
        else:
            child_arch[position] = 0
    #随机插入一个P操作
    elif mutate_type == 1:
        position = random.randint(0, length)
        new_value = random.randint(2, 4)
        child_arch.insert(position, new_value)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length)
        child_arch.insert(position, 0)
    return child_arch


def mutate_arch_xiaorong(parent_arch):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    position = random.randint(0, length-1)
    child_arch[position] ^= 1
    return child_arch