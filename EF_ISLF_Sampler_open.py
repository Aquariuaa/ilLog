import numpy as np
import random
from collections import Counter

def reduce_repeats(dataset):
    reduce_set = set()
    for t in dataset:
        reduce_set.add(tuple(t))
    print("len(reduce_set):", len(reduce_set))
    return list(reduce_set)


def ilsf_processing(set):
    datas = np.array(set)[:,:-1]
    flatten = datas.reshape(-1)
    types = list(Counter(flatten.tolist()))
    # print("the type of events is ", types)
    sequences = len(datas)
    # print("the number of sequences is ", sequences)
    ilsf = {}
    for type in types:
        count = 0
        for data in datas:
            if type in data:
                count += 1
        ilsf.update({type:np.log(sequences/count)/np.log(2)})
    return ilsf

def ef(set, ilsf, memory):
    set = np.array(set)
    datas = set[:,:-1]
    eis = []
    for i in range(0,len(datas)):
        statistics = Counter(datas[i].tolist())
        ei = 0
        for key, value in statistics.items():
            ei = ei + ilsf[key]*value
        eis.append(ei)
    eis = np.array(eis)
    sort = eis.argsort()
    # print("memory:", memory)
    samples = set[sort[-memory:]]
    # print("len_sample",len(samples))
    return samples


def ef_ilsf(set, memory):
    set = reduce_repeats(np.array(set))
    # set = np.array(set)
    normal_set = []
    abnormal_set = []
    for log in set:
        if log[-1] == 0:
            normal_set.append(log)
        else:
            abnormal_set.append(log)

    ilsf_normal = ilsf_processing(normal_set)
    ilsf_abnormal = ilsf_processing(abnormal_set)
    normal = ef(normal_set, ilsf_normal, memory)
    abnormal = ef(abnormal_set, ilsf_abnormal, memory)
    return np.concatenate((normal,abnormal),axis=0)

def counter_logs(set):
    # print("len(set):", len(set))
    # print("len(set[0]), len(set[1])",len(set[0]), len(set[1]))
    print("len(np.array(set[0]))",len(np.array(set[0])))
    # print("len(reduce_repeats(np.array(set[0])))",len(reduce_repeats(np.array(set[0]))))
    print("len(np.array(set[1]))",len(np.array(set[1])))
    # print("len(reduce_repeats(np.array(set[1])))",len(reduce_repeats(np.array(set[1]))))




def construct_efilsf(set, idx, memory, base_memory):
    memory = int(memory/2)
    base_memory = int(base_memory/2)
    counter_logs(set)
    base_task= ef_ilsf(set[0], base_memory)
    for i in range(idx-1):
        base_task = np.vstack((ef_ilsf(set[i+1], memory), base_task))
    base_task = np.vstack((reduce_repeats(np.array(set[idx])), base_task))
    base_task = np.vstack((np.array(set[idx]), base_task))
    return base_task.tolist()
