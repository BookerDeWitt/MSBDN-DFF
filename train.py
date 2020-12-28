from __future__ import print_function
import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataSet_HDF5
from importlib import import_module
import random
import re
import time
import statistics
import torch.nn.functional as F


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument('--dataset', default="/data/Datasets/RESIDE/RESIDE_HDF5_all/", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--model', default='MSBDN-DFF', type=str, help='Import which network')
parser.add_argument('--name', default='MSBDN-DFF', type=str, help='Filename of the training models')
parser.add_argument('--gpu_ids', type=str, default='0, 1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--train_step", type=int, default=1, help="Activated gate module")
parser.add_argument("--clip", type=float, default=0.25, help="Clipping Gradients. Default=0.1")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")

training_settings=[
    {'nEpochs': 100, 'lr': 1e-4, 'step': 50, 'lr_decay': 0.1}
]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def mkdir_steptraing():
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    step1_folder, step2_folder, step3_folder, step4_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3'), join(models_folder, '4')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        print("===> Step training models store in models/1 & /2 & /3.")

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])

def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[-3:-2])
    start_epoch = "".join(re.findall(r"\d", resume)[-2:])
    return int(trainingstep), int(start_epoch)+1

def adjust_learning_rate(epoch):
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def checkpoint(step, epoch):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    model_out_path = join(models_folder, "{0}/MSBDN_epoch_{1:02d}.pkl".format(step, epoch))
    torch.save(model, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, criterion, optimizer, epoch):

    epoch_loss = 0
    start_time_data=0
    med_time_data = []
    med_time_gpu = []

    for iteration, batch in enumerate(train_gen, 1):
        evalation_time_data = time.perf_counter() - start_time_data
        med_time_data.append(evalation_time_data)
        start_time_gpu = time.perf_counter()

        Hazy = batch[0]
        GT = batch[1]
        Hazy = Hazy.to(device)
        GT = GT.to(device)

        dehaze = model(Hazy)
        mse = criterion(dehaze, GT)

        epoch_loss += mse
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        evalation_time_gpu = time.perf_counter() - start_time_gpu
        med_time_gpu.append(evalation_time_gpu)

        if iteration % 100 == 0:
            median_time_data = statistics.median(med_time_data)
            median_time_gpu = statistics.median(med_time_gpu)
            print("===> Loading Time: {:.6f}; Runing Time:{:.6f}".format(median_time_data, median_time_gpu))
            print("===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), mse.cpu()))
            med_time_data = []
            med_time_gpu = []
        start_time_data = time.perf_counter()
    print("===>Epoch{} Part: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))
    return epoch_loss / len(trainloader)


opt = parser.parse_args()
Net = import_module('networks.' + opt.model)
print(opt.resume)
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
str_ids = opt.gpu_ids.split(',')
torch.cuda.set_device(int(str_ids[0]))
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

train_dir = opt.dataset
train_sets = [x for x in sorted(os.listdir(train_dir)) if is_hdf5_file(x)]
print("===> Loading model {} and criterion".format(opt.model))

if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model = Net.make_model(opt)
        model_dict = model.state_dict()
        print(get_n_params(model))
        pretrained_model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(get_n_params(model))
        opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)
        print(opt.start_training_step)
        print(opt.start_epoch)
        mkdir_steptraing()
else:
    model = Net.make_model(opt)
    print(get_n_params(model))
    mkdir_steptraing()


model = model.to("cuda")
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for i in range(opt.start_training_step, 3):
    opt.nEpochs   = training_settings[i - 1]['nEpochs']
    opt.lr        = training_settings[i - 1]['lr']
    opt.step      = training_settings[i - 1]['step']
    opt.lr_decay  = training_settings[i - 1]['lr_decay']
    print(opt)
    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        psnr = 0
        adjust_learning_rate(epoch)
        random.shuffle(train_sets)
        for j in range(len(train_sets)):
            print("Step {}:Training folder is {}".format(i, join(train_dir, train_sets[j])))
            train_set = DataSet_HDF5(join(train_dir, train_sets[j]))
            trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=1)
            avg_psnr = train(trainloader, model, criterion, optimizer, epoch)
            psnr = psnr + avg_psnr
        if epoch % 1 == 0:
            checkpoint(i, epoch)
        psnr = psnr / len(train_sets)
        print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, psnr))
        with open("Logs/Output_{}.txt".format(opt.name), "a+") as text_file:
            print("===>Epoch{} Complete: Avg loss is :{:4f}\n".format(epoch, psnr), file=text_file)

    opt.start_epoch = 1
