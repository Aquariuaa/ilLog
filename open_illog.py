import json
import random
import torch
import numpy as np
import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from classifier import NegLayer
from EF_ISLF_Sampler import construct_efilsf
from Random_Sampler import construct_random
import openturns as ot

args = {
    "increment": 5,
    "Model": "Model()",
    "dataset": "tbird",
    "memory": 7000,
    "base_memory":7000,
    "base": 0,
    "epochs": 20,
    "dataset_split": 0.7,
    "eval": "True",
    "data_path": "/home/lab/Workspaces/**/data/",
    "window_size": 20,
    "gen_dim": 21,
    "lr":1.0e-4,
    # esr-->efilsf
    "sampler":"efilsf",
    "margin":0,
    "scale":1,
    "L":200,
    "method":"SCP"
}

# Read raw log
def data_read(filepath):
    fp = open(filepath, "r")
    datas = []
    lines = fp.readlines()
    for line in tqdm.tqdm(lines):
        row = line.strip()
        datas.append(row)
    fp.close()
    return datas

def task_processing_base(num_task, base_split_num):
    dataset = args["dataset"]
    data_path = args["data_path"]
    window_size = args["window_size"]
    if "hdfs" in dataset:
        path_normal = data_path + dataset + "_test_normal"
        path_abnormal = data_path + dataset + "_test_abnormal"
    else:
        path_normal = data_path + dataset + "_time_normal_" + str(window_size)
        path_abnormal = data_path + dataset + "_time_abnormal_" + str(window_size)

    data_normal = data_read(path_normal)
    data_abnormal = data_read(path_abnormal)
    label_normal = []
    label_abnormal = []
    for i in range(0,len(data_normal)):
        label_normal.append(0)
    for j in range(0,len(data_abnormal)):
        label_abnormal.append(1)

    base_block_normal = int(len(data_normal)*base_split_num)
    base_block_abnormal = int(len(data_abnormal)*base_split_num)

    data_normal_base = data_normal[:base_block_normal]
    label_normal_base = label_normal[:base_block_normal]
    data_abnormal_base = data_abnormal[:base_block_abnormal]
    label_abnormal_base = label_abnormal[:base_block_abnormal]
    base_data = data_normal_base + data_abnormal_base
    base_label = label_normal_base + label_abnormal_base
    base_set = [list(t) for t in zip(base_data, base_label)]

    data_normal = data_normal[base_block_normal:]
    label_normal = label_normal[base_block_normal:]
    # 此处出现了一个错误
    data_abnormal = data_abnormal[base_block_abnormal:]
    label_abnormal = label_abnormal[base_block_abnormal:]
    block_normal = int(len(data_normal) / num_task)
    block_abnormal = int(len(data_abnormal) / num_task)

    task_datas = []
    task_datas.append(base_set)
    for num in range(num_task):
        if num <num_task-1:
            sample_datas = data_normal[num*block_normal:(num+1)*block_normal] + data_abnormal[num*block_abnormal:(num+1)*block_abnormal]
            sample_labels = label_normal[num*block_normal:(num+1)*block_normal] + label_abnormal[num*block_abnormal:(num+1)*block_abnormal]
            task_datas.append([list(t) for t in zip(sample_datas, sample_labels)])
        else:
            sample_datas = data_normal[num*block_normal:] + data_abnormal[num*block_abnormal:]
            sample_labels = label_normal[num*block_normal:] + label_abnormal[num*block_abnormal:]
            task_datas.append([list(t) for t in zip(sample_datas, sample_labels)])
    return task_datas


# Processing Dataset by task num
def task_processing(num_task):
    dataset = args["dataset"]
    data_path = args["data_path"]
    window_size = args["window_size"]
    if "hdfs" in dataset:
        path_normal = data_path + dataset + "_test_normal"
        path_abnormal = data_path + dataset + "_test_abnormal"
    else:
        path_normal = data_path + dataset + "_time_normal_" + str(window_size)
        path_abnormal = data_path + dataset + "_time_abnormal_" + str(window_size)

    data_normal = data_read(path_normal)
    data_abnormal = data_read(path_abnormal)
    label_normal = []
    label_abnormal = []
    for i in range(0,len(data_normal)):
        label_normal.append(0)
    for j in range(0,len(data_abnormal)):
        label_abnormal.append(1)
    block_normal = int(len(data_normal) / num_task)
    block_abnormal = int(len(data_abnormal) / num_task)

    task_datas = []
    for num in range(num_task):
        if num <num_task-1:
            sample_datas = data_normal[num*block_normal:(num+1)*block_normal] + data_abnormal[num*block_abnormal:(num+1)*block_abnormal]
            sample_labels = label_normal[num*block_normal:(num+1)*block_normal] + label_abnormal[num*block_abnormal:(num+1)*block_abnormal]
            task_datas.append([list(t) for t in zip(sample_datas,sample_labels)])
        else:
            sample_datas = data_normal[num*block_normal:] + data_abnormal[num*block_abnormal:]
            sample_labels = label_normal[num*block_normal:] + label_abnormal[num*block_abnormal:]
            task_datas.append([list(t) for t in zip(sample_datas,sample_labels)])
    return task_datas

# Read .json for Semantic
def semantic_json(filename):
    with open(filename) as f:
        gdp_list = json.load(f)
        value = list(gdp_list.values())
    return np.array(value)

def bulid_dataloader(task_data):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(task_data)
    x, y = torch.from_numpy(norm_data[:, 0:-1]), torch.from_numpy(norm_data[:, -1])
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    x = x.view(-1,1,args["window_size"])
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=128, shuffle=True, drop_last=False)
    return data_loader

def dataset_processing_eval(task_data, split):
    random.seed(100)
    random.shuffle(task_data)
    normal = []
    abnormal = []
    for i in range(0, len(task_data)):
        if task_data[i][1] == 0:
            normal.append(task_data[i])
        else:
            abnormal.append(task_data[i])

    split_len = int(len(abnormal)*split)
    train = normal[:split_len] + abnormal[:split_len]
    test = normal[split_len:] + abnormal[split_len:]

    task_train = np.zeros((len(train), args["gen_dim"]))
    task_test = np.zeros((len(test), args["gen_dim"]))
    for i in range(0, len(train)):
        data = [int(n) for n in train[i][0].split()]
        task_train[i][:len(data)] = np.array(data)
        task_train[i][-1] = train[i][1]

    for j in range(0, len(test)):
        data = [int(n) for n in test[j][0].split()]
        task_test[j][:len(data)] = np.array(data)
        task_test[j][-1] = test[j][1]

    task_train = task_train.tolist()
    task_test = task_test.tolist()

    return task_train, task_test


def dataset_processing(task_data, split):
    random.seed(100)
    random.shuffle(task_data)
    result_data = np.zeros((len(task_data), args["gen_dim"]))
    for i in range(0, len(task_data)):
        data = [int(n) for n in task_data[i][0].split()]
        result_data[i][:len(data)] = np.array(data)
        result_data[i][-1] = task_data[i][1]
    result_data = result_data.tolist()
    task_train = np.array(result_data[:int(len(result_data) * split)])
    task_test = np.array(result_data[int(len(result_data) * split):])
    return task_train, task_test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args["base"] == 0:
    task_datas = task_processing(args["increment"])
else:
    task_datas = task_processing_base(args["increment"]-1, base_split_num= args["base"])
task_trains = []
task_tests = []
for task_data in task_datas:
    if args["eval"]:
        task_train, task_test = dataset_processing_eval(task_data, split=args["dataset_split"])
    else:
        task_train, task_test = dataset_processing(task_data, split=args["dataset_split"])
    task_trains.append(task_train)
    task_tests.append(task_test)
train_dataloaders = [bulid_dataloader(task) for task in task_trains]
test_dataloaders = [bulid_dataloader(task) for task in task_tests]



class Model(nn.Module):
  """
  Model architecture
  1*28*28 (input) → 1024 → 512 → 256 → 10
  """
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(args["window_size"], 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 128)
    self.classifier = NegLayer(128, 2, args["margin"], args["scale"])
    self.relu = nn.ReLU()


  def forward(self, x, label=None):
    x = x.view(-1, args["window_size"])
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    x = self.classifier(x, label)
    return x

def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device, tdls,log_step=1):
    model.train()
    model.zero_grad()
    objective = nn.CrossEntropyLoss()
    acc_per_epoch = []
    bar = epochs_per_task
    print("**********************************************************************************")
    f1_records = np.zeros((args["epochs"], args["increment"]))
    for epoch in range(bar):
        total = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs, labels.long())
            loss = objective(outputs, labels.long())
            total_loss = loss
            lll_loss = lll_object.penalty(model)
            total_loss += lll_lambda * lll_loss
            lll_object.update(model)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss = total_loss.item()
            total+=len(imgs)
        acc_average = []
        for test_dataloader in test_dataloaders:
            acc_test = evaluate(model, test_dataloader, device)
            acc_average.append(acc_test)
        average = np.mean(np.array(acc_average))
        pre_results,rec_results,f1_results = evaluate_2(model, test_dataloaders, device)
        f1_records[epoch, :len(test_dataloaders)] = f1_results

    acc_per_epoch.append(average * 100.0)
    pre_results, rec_results, f1_results = evaluate_(model, tdls, device)
    return model, optimizer, acc_per_epoch, pre_results, rec_results, f1_results, f1_records



def evaluate_(model, test_dataloaders, device):
    model.eval()
    correct_cnt = 0
    total = 0
    pre_results = []
    rec_results = []
    f1_results = []
    ave_pre, ave_rec, ave_f1 = 0, 0, 0
    for l in range(len(test_dataloaders)):
        test_dataloader = test_dataloaders[l]
        tp, fp, tn, fn = 0, 0, 0, 0
        for logs, labels in test_dataloader:
            logs, labels = logs.to(device), labels.to(device)
            outputs = model(logs)
            _, pred_label = torch.max(outputs.data, 1)
            correct_cnt += (pred_label == labels.data).sum().item()
            total += torch.ones_like(labels.data).sum().item()
            y_pred = pred_label.tolist()
            label = labels.tolist()
            for j in range(0, len(y_pred)):
                if y_pred[j] == label[j] and label[j] == 0:
                    tp = tp + 1
                elif y_pred[j] != label[j] and label[j] == 0:
                    fn = fn + 1
                elif y_pred[j] == label[j] and label[j] == 1:
                    tn = tn + 1
                elif y_pred[j] != label[j] and label[j] == 1:
                    fp = fp + 1
        print(tp, fp, tn, fn)
        pre = round(tp / ((tp + fp) + 0.0001) * 100, 1)
        rec = round(tp / ((tp + fn) + 0.0001) * 100, 1)
        f1 = round((2 * pre * rec) / ((pre + rec) + 0.0001), 1)
        ave_pre = ave_pre + pre
        ave_rec = ave_rec + rec
        ave_f1 = ave_f1 + f1
        pre_results.append(pre)
        rec_results.append(rec)
        f1_results.append(f1)
    pre_results.append(round(ave_pre/len(test_dataloaders),1))
    rec_results.append(round(ave_rec/len(test_dataloaders),1))
    f1_results.append(round(ave_f1/len(test_dataloaders),1))
    return pre_results,rec_results,f1_results

def evaluate_2(model, test_dataloaders, device):
    model.eval()
    correct_cnt = 0
    total = 0
    pre_results = []
    rec_results = []
    f1_results = []
    ave_pre, ave_rec, ave_f1 = 0, 0, 0
    for l in range(len(test_dataloaders)):
        test_dataloader = test_dataloaders[l]
        tp, fp, tn, fn = 0, 0, 0, 0
        for logs, labels in test_dataloader:
            logs, labels = logs.to(device), labels.to(device)
            outputs = model(logs)
            _, pred_label = torch.max(outputs.data, 1)
            correct_cnt += (pred_label == labels.data).sum().item()
            total += torch.ones_like(labels.data).sum().item()
            y_pred = pred_label.tolist()
            label = labels.tolist()
            for j in range(0, len(y_pred)):
                if y_pred[j] == label[j] and label[j] == 0:
                    tp = tp + 1
                elif y_pred[j] != label[j] and label[j] == 0:
                    fn = fn + 1
                elif y_pred[j] == label[j] and label[j] == 1:
                    tn = tn + 1
                elif y_pred[j] != label[j] and label[j] == 1:
                    fp = fp + 1
        # print(tp, fp, tn, fn)
        pre = round(tp / ((tp + fp) + 0.0001) * 100, 1)
        rec = round(tp / ((tp + fn) + 0.0001) * 100, 1)
        f1 = round((2 * pre * rec) / ((pre + rec) + 0.0001), 1)
        acc = round((tp + tn) / (tp + fn + tn + fp) * 100, 1)
        pre_results.append(pre)
        rec_results.append(rec)
        f1_results.append(f1)
    return pre_results,rec_results,f1_results


def evaluate(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total = 0
    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)
        correct_cnt += (pred_label == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()
    return correct_cnt / total

def sample_spherical(npoints, ndim=3):
    sequence = ot.LowDiscrepancySequence(ot.HaltonSequence(npoints))
    halton_data = sequence.generate(ndim)
    vec = np.array(halton_data)
    vec /= np.linalg.norm(vec, axis=0)
    # print("vec.shape2", vec.shape)
    return torch.from_numpy(vec)


# qmc-scp
class scp(object):
    def __init__(self, model: nn.Module, dataloader, L: int, device, prev_guards=[None]):
        self.model = model
        self.dataloader = dataloader
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._state_parameters = {}
        self.L = L
        self.device = device
        self.previous_guards_list = prev_guards
        self._precision_matrices = self.calculate_importance()
        for n, p in self.params.items():
            self._state_parameters[n] = p.clone().detach()

    def calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)
            for i in range(len(self.previous_guards_list)):
                if self.previous_guards_list[i]:
                    precision_matrices[n] += self.previous_guards_list[i][n]

        self.model.eval()
        if self.dataloader is not None:
            num_data = len(self.dataloader)
            for data in self.dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))
                # print("output.shape:", output.shape) 100, torch.Size([128, 2])
                mean_vec = output.mean(dim=0)
                # print("mean_vec.shape:", mean_vec.shape)
                L_vectors = sample_spherical(self.L, output.shape[-1])
                # print("L_vectors.shape:", L_vectors.shape)
                L_vectors = L_vectors.transpose(1, 0).to(self.device).float()
                # print("L_vectors.shape2:", L_vectors.shape)
                total_scalar = 0
                for vec in L_vectors:
                    scalar = torch.matmul(vec, mean_vec)
                    total_scalar += scalar
                total_scalar /= L_vectors.shape[0]
                total_scalar.backward()
                for n, p in self.model.named_parameters():
                    precision_matrices[n].data += p.grad ** 2 / num_data

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._state_parameters[n]) ** 2
            loss += _loss.sum()
        return loss

    def update(self, model):
        # do nothing
        return


# Baseline
print("RUN BASELINE")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
lll_object = scp(model=model, dataloader=None, L=args["L"], device=device)
lll_lambda = 100
scp_acc = []
task_bar = tqdm.auto.trange(len(train_dataloaders), desc="Task   1")
prev_guards = []

pres1 = []
recs1 = []
f1s1 = []
curve_recode = np.zeros((args["increment"]*args["epochs"], args["increment"]))
for train_indexes in task_bar:
    # Train each task
    if train_indexes  ==  0:
        model, _, acc_list, pre1, rec1, f11, f1_records = train(model, optimizer, train_dataloaders[train_indexes],  args["epochs"],
                                   lll_object, lll_lambda, evaluate=evaluate, device=device,
                                   test_dataloaders=test_dataloaders[:train_indexes + 1],
                                                 tdls=test_dataloaders)
    else:
        if args["sampler"] == "efilsf":
            updata_data = construct_efilsf(task_trains,train_indexes, args["memory"], args["base_memory"])
            updata_data_loader = bulid_dataloader(updata_data)
        elif args["sampler"] == "random":
            updata_data = construct_random(task_trains,train_indexes, args["memory"], args["base_memory"])
            updata_data_loader = bulid_dataloader(updata_data)
        elif args["sampler"] == "baseline":
            updata_data_loader = train_dataloaders[train_indexes]
        else:
            print("sampler is wrong!")
        model, _, acc_list, pre1, rec1, f11, f1_records = train(model, optimizer, updata_data_loader, args["epochs"],
                                   lll_object, lll_lambda, evaluate=evaluate, device=device,
                                   test_dataloaders=test_dataloaders[:train_indexes + 1],
                                                 tdls=test_dataloaders)

    pres1.append(pre1)
    recs1.append(rec1)
    f1s1.append(f11)
    curve_recode[args["epochs"] * (train_indexes):args["epochs"] * (train_indexes + 1)] = f1_records
    # get model weight to baseline class and do nothing!
    lll_object = scp(model=model, dataloader=train_dataloaders[train_indexes], L=args["L"], device=device,prev_guards=prev_guards)
    # new a optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    # Collect average accuracy in each epoch
    scp_acc.extend(acc_list)
    # display the information of the next task.
    task_bar.set_description_str(f"Task  {train_indexes + 2:2}")

def compute_mean(pres,recs,f1s):
    pres = np.array(pres)
    recs = np.array(recs)
    f1s = np.array(f1s)
    pres = np.around(np.vstack((pres,pres.mean(axis=0))),2)
    recs = np.around(np.vstack((recs,recs.mean(axis=0))),2)
    f1s = np.around(np.vstack((f1s,f1s.mean(axis=0))),2)
    return pres,recs,f1s

pres1, recs1, f1s1 = compute_mean(pres1, recs1, f1s1)
pre_con = pres1[-1]
rec_con = recs1[-1]
f1_con = f1s1[-1]
print(f1_con)