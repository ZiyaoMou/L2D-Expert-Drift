import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import time
import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from common.utils import AverageMeter, accuracy, metrics_print
from common.model import WideResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_CIFAR10_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_default_expert = os.path.join(_CIFAR10_ROOT, "expert", "curve_50.csv")
exp_curve = pd.read_csv(os.environ.get("CIFAR10_EXPERT_CURVE", _default_expert))
p_curve = exp_curve['p']


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]  # batch_size
    rc = [n_classes] * batch_size
    outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
        outputs[range(batch_size), labels])  
    return torch.sum(outputs) / batch_size

def my_CrossEntropyLoss(outputs, labels):
    # Regular Cross entropy loss
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels])  # regular CE
    return torch.sum(outputs) / batch_size

def train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha):
    """Train for one epoch on the training set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model = model.to(device)
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        output = model(input)
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(input, target)
        m2 = [0] * batch_size
        for j in range(0, batch_size):
            if m[j] == target[j].item():
                m[j] = 1
                m2[j] = alpha
            else:
                m[j] = 0
                m2[j] = 1
        m = torch.tensor(m)
        m2 = torch.tensor(m2)
        m = m.to(device)
        m2 = m2.to(device)
        # done getting expert predictions and costs 
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

def train_reject_class(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha):
    """Train for one epoch on the training set without deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)

        # compute loss
        loss = my_CrossEntropyLoss(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))


def validate_reject(val_loader, model, epoch, expert_fn, n_classes):
    """Perform validation on the validation set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
        # expert prediction
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(input, target)
        alpha = 1
        m2 = [0] * batch_size
        for j in range(0, batch_size):
            if m[j] == target[j].item():
                m[j] = 1
                m2[j] = alpha
            else:
                m[j] = 0
                m2[j] = 1
        m = torch.tensor(m)
        m2 = torch.tensor(m2)
        m = m.to(device)
        m2 = m2.to(device)
        # compute loss
        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
def run_reject(model, data_aug, n_dataset, expert_fn, epochs, alpha):
    '''
    model: WideResNet model
    data_aug: boolean to use data augmentation in training
    n_dataset: number of classes
    expert_fn: expert model
    epochs: number of epochs to train
    alpha: alpha parameter in L_{CE}^{\alpha}
    '''
    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if n_dataset == 10:
        dataset = 'cifar10'
    elif n_dataset == 100:
        dataset = 'cifar100'

    kwargs = {'num_workers': 4, 'pin_memory': True}


    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                           transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    test_size = len(train_dataset_all) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print(len(train_loader))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    for epoch in range(0, epochs):
        # train for one epoch
        train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_dataset, alpha)
        if epoch % 10 == 0:
            metrics_print(model, expert_fn, n_dataset, test_loader)


# k = 5

n_dataset = 10  # cifar-10

class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() <= self.k:
                outs[i] = labels[i].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs
class SeqExpert:
    def __init__(
        self,
        n_classes: int,
        acc_curve,
        other_acc,
        k: int,
        cycle: bool = True,
        seed: int = 66,
    ):
        self.n_classes = int(n_classes)
        self.k = int(k)
        self.acc_curve = np.asarray(acc_curve, dtype=float).clip(0.0, 1.0)
        self.T = len(self.acc_curve)
        self.t = 0
        self.cycle = cycle

        if other_acc is None:
            self.other_acc_curve = np.full(self.T, 1.0 / self.n_classes, dtype=float)
        else:
            arr = np.asarray(other_acc, dtype=float)
            if arr.ndim == 0:
                self.other_acc_curve = np.full(self.T, float(arr), dtype=float)
            else:
                assert len(arr) == self.T, "other_acc_curve length must match acc_curve"
                self.other_acc_curve = arr.clip(0.0, 1.0)

        self.rng = np.random.default_rng(seed)

    def _next_indices(self, batch_size: int):
        if self.cycle:
            idx = (np.arange(batch_size) + self.t) % self.T
            self.t = (self.t + batch_size) % self.T
        else:
            idx = np.minimum(np.arange(batch_size) + self.t, self.T - 1)
            self.t = min(self.t + batch_size, self.T - 1)
        return idx

    @torch.no_grad()
    def predict(self, inputs, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B,) long on any device
        device = labels.device
        B = labels.size(0)
        idx = self._next_indices(B)

        p_master = self.acc_curve[idx]            # (B,)
        p_other  = self.other_acc_curve[idx]      # (B,)

        y_true = labels.detach().long().cpu().numpy()  # (B,)
        mask_master = (y_true < self.k)

        u = self.rng.random(B)
        hit = np.empty(B, dtype=bool)
        hit[mask_master] = (u[mask_master] < p_master[mask_master])
        hit[~mask_master] = (u[~mask_master] < p_other[~mask_master])

        def sample_wrong(y_np):
            r = self.rng.integers(0, self.n_classes - 1, size=y_np.shape)
            wrong = r + (r >= y_np)
            return wrong

        preds = np.empty(B, dtype=np.int64)
        preds[hit]  = y_true[hit]
        preds[~hit] = sample_wrong(y_true[~hit])

        return torch.from_numpy(preds).to(device=device, dtype=torch.long)

def build_window_experts(p_curve, p_low_curve, k=5, n_classes=10, num_windows=10, seed_base=66):
    pc = np.asarray(p_curve, dtype=float)
    pl = np.asarray(p_low_curve, dtype=float)
    L = len(pc)

    experts = []
    for w in range(num_windows):
        start = int(np.floor(w * L / num_windows))
        end   = int(np.floor((w + 1) * L / num_windows)) if w < num_windows - 1 else L

        acc_win = pc[start:end]
        oth_win = pl[start:end]

        exp_w = SeqExpert(
            n_classes=n_classes,
            acc_curve=acc_win,
            other_acc=oth_win,
            k=k,
            cycle=True,
            seed=seed_base + w,
        )
        experts.append(exp_w)
    return experts


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--seeds", type=int, default=None)
args = parser.parse_args()

run_seeds = []

if args.seeds is not None:
    run_seeds = [int(s) for s in args.seeds.split(",") if s.strip()!=""]
elif args.seed is not None:
    run_seeds = [args.seed]
else:
    run_seeds = [42, 43, 44, 45, 46]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_windows = 10
epochs_each = 200
alpha = 1
n_classes = n_dataset

T = 50
p_curve = np.array(p_curve)
p_low_curve = p_curve
# gap = len(p_curve) // T
# p_curve_new = []
# for i in range(T):
#     start = i * gap
#     end = min((i + 1) * gap, len(p_curve))
#     segment = p_curve[start:end]
#     avg = np.mean(segment)
#     p_curve_new.append(avg)

# p_curve = np.array(p_curve_new)
# # p_curve = np.linspace(1, 0.5, T)
# p_low_curve = np.full(T, 0.1)
# print(len(p_curve))
# print(len(p_low_curve))

timestamp = time.strftime("%Y%m%d_%H%M%S")
base_save_dir = os.path.join(_CIFAR10_ROOT, "models", "perstep_curve_gaussian_k10")
os.makedirs(base_save_dir, exist_ok=True)

for seed in run_seeds:
    print(f"\n================ SEED {seed} ================\n")
    set_seed(seed)
    try:
        experts = build_window_experts(
            p_curve, p_low_curve, k=10, n_classes=n_classes,
            num_windows=num_windows, seed=seed
        )
    except TypeError:
        experts = build_window_experts(
            p_curve, p_low_curve, k=10, n_classes=n_classes,
            num_windows=num_windows
        )

    save_dir = os.path.join(base_save_dir, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    for w_id, expert in enumerate(experts):
        print(f"\n======== Training model for window {w_id} (seed={seed}) ========")
        model = WideResNet(28, n_classes + 1, 4, dropRate=0.0)

        run_reject(
                model=model,
                data_aug=False,
                n_dataset=n_classes,
                expert_fn=expert.predict,
                epochs=epochs_each,
                alpha=alpha,
            )

        out_path = os.path.join(save_dir, f"l2d_window_{w_id:02d}.pth")
        torch.save(model, out_path)
        print(f"Saved -> {out_path}")