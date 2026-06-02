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

import time
import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from common.utils import AverageMeter, accuracy
from common.model import WideResNet

_CIFAR10_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_default_expert = os.path.join(_CIFAR10_ROOT, "expert", "curve_50.csv")
exp_curve = pd.read_csv(os.environ.get("CIFAR10_EXPERT_CURVE", _default_expert))
p_curve = exp_curve['p']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.data import Dataset
from typing import Tuple, Any

class IndexedCIFAR10(Dataset):
    def __init__(self, original_dataset: Dataset, seq_len: int):
        self.original_dataset = original_dataset
        self.seq_len = seq_len
        self.num_samples = len(original_dataset)
        
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int, int]:
        img, label = self.original_dataset[idx]
        seq_id = idx // self.seq_len
        t = idx % self.seq_len
        return img, label, seq_id, t

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
    model.train()

    end = time.time()
    for i, (input, target, seq_ids, ts) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        ts = ts.to(device)

        output = model(input)

        batch_size = output.size()[0]
        m = expert_fn(input, target, ts)
        
        m2 = [0] * batch_size
        for j in range(batch_size):
            if m[j] == target[j].item():
                m[j] = 1
                m2[j] = alpha
            else:
                m[j] = 0
                m2[j] = 1
        m = torch.tensor(m).to(device)
        m2 = torch.tensor(m2).to(device)
        
        loss = reject_CrossEntropyLoss(output, m, target, m2, n_classes)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

def metrics_print(net, expert_fn, n_classes, loader, T, save_metrics_csv=None):
    """
    Evaluates model performance on a sequential dataset.

    Args:
        net: The neural network model (e.g., WideResNet).
        expert_fn: The function that generates expert predictions.
        n_classes: The number of classification classes.
        loader: The data loader, which provides (images, labels, seq_ids, ts).
        T: The length of each sequence.
    """
    net.eval()
    
    # Store metrics for each time step 't'
    step_metrics = [{
        "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
        "real_total": 0, "exp_total": 0, "cls_total": 0
    } for _ in range(T)]

    # Store overall metrics
    overall_metrics = {
        "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
        "real_total": 0, "exp_total": 0, "cls_total": 0
    }

    with torch.no_grad():
        for images, labels, seq_ids, ts in loader:
            images = images.to(device)
            labels = labels.to(device)
            ts = ts.to(device)
            batch_size = images.size(0)

            # Get model and expert outputs
            outputs = net(images)
            exp_predictions = expert_fn(images, labels, ts)
            _, predicted = torch.max(outputs.data, 1)

            # Process each sample in the batch
            for i in range(batch_size):
                t = ts[i].item() # Get the time step for the current sample
                
                # Check if the model has a deferral class (n_classes)
                # Your old code checks for `predicted[i] == 10`.
                # If your WideResNet has n_classes + 1 outputs, `10` is the deferral class.
                is_deferred = (predicted[i].item() == n_classes)
                
                # Get the classifier's prediction (even if it defers)
                alone_pred = predicted[i]
                if is_deferred:
                    # If deferred, the classifier's 'final' prediction
                    # is the second highest confidence among the classification classes.
                    # Your original logic is flawed here; a simpler approach is to
                    # treat the highest confidence among the n_classes as the prediction.
                    # For simplicity and correctness with WideResNet, let's just get
                    # the argmax over the classification classes (excluding the deferral).
                    alone_pred = torch.max(outputs.data[i, :n_classes], 0)[1]

                # Accumulate metrics for the current time step and overall
                metrics = step_metrics[t]
                
                # alone_correct: The accuracy of the classifier alone (never defers)
                is_alone_correct = (alone_pred == labels[i]).item()
                metrics["alone_correct"] += is_alone_correct
                overall_metrics["alone_correct"] += is_alone_correct

                if not is_deferred:
                    # Classifier's decision
                    is_cls_correct = (predicted[i] == labels[i]).item()
                    metrics["correct_cls"] += is_cls_correct
                    metrics["cls_total"] += 1
                    metrics["correct_sys"] += is_cls_correct
                    
                    overall_metrics["correct_cls"] += is_cls_correct
                    overall_metrics["cls_total"] += 1
                    overall_metrics["correct_sys"] += is_cls_correct
                else:
                    # Expert's decision
                    is_exp_correct = (exp_predictions[i] == labels[i].item())
                    metrics["exp"] += is_exp_correct
                    metrics["exp_total"] += 1
                    metrics["correct_sys"] += is_exp_correct
                    
                    overall_metrics["exp"] += is_exp_correct
                    overall_metrics["exp_total"] += 1
                    overall_metrics["correct_sys"] += is_exp_correct
                    
                metrics["real_total"] += 1
                overall_metrics["real_total"] += 1

    save_metrics = []
    # Print per-time-step metrics
    print("\n--- Per-Time-Step Metrics ---")
    for t in range(T):
        metrics = step_metrics[t]
        if metrics["real_total"] > 0:
            coverage = metrics["cls_total"] / metrics["real_total"]
            system_acc = 100 * metrics["correct_sys"] / metrics["real_total"]
            expert_acc = 100 * metrics["exp"] / (metrics["exp_total"] + 1e-5)
            classifier_acc = 100 * metrics["correct_cls"] / (metrics["cls_total"] + 1e-5)
            alone_classifier_acc = 100 * metrics["alone_correct"] / metrics["real_total"]
            
            print(f"\n--- Metrics for Time Step {t+1} ---")
            print(f"coverage: {coverage:.4f}")
            print(f"system accuracy: {system_acc:.3f}")
            print(f"expert accuracy: {expert_acc:.3f}")
            print(f"classifier accuracy: {classifier_acc:.3f}")
            print(f"alone classifier: {alone_classifier_acc:.3f}")
            if save_metrics_csv is not None:
                save_metrics.append({
                    "time_step": t+1,
                    "coverage": coverage,
                    "system_acc": system_acc,
                    "expert_acc": expert_acc,
                    "classifier_acc": classifier_acc,
                    "alone_classifier": alone_classifier_acc
                })


    # Print overall metrics
    print("\n--- Overall Metrics (Average over all time steps) ---")
    if overall_metrics["real_total"] > 0:
        overall_coverage = overall_metrics["cls_total"] / overall_metrics["real_total"]
        overall_system_acc = 100 * overall_metrics["correct_sys"] / overall_metrics["real_total"]
        overall_expert_acc = 100 * overall_metrics["exp"] / (overall_metrics["exp_total"] + 1e-5)
        overall_classifier_acc = 100 * overall_metrics["correct_cls"] / (overall_metrics["cls_total"] + 1e-5)
        overall_alone_classifier_acc = 100 * overall_metrics["alone_correct"] / overall_metrics["real_total"]
        
        to_print = {
            "coverage": f"{overall_metrics['cls_total']} out of {overall_metrics['real_total']}",
            "system accuracy": overall_system_acc,
            "expert accuracy": overall_expert_acc,
            "classifier accuracy": overall_classifier_acc,
            "alone classifier": overall_alone_classifier_acc
        }
        if save_metrics_csv is not None:
            save_metrics.append({
                "time_step": "overall",
                "coverage": overall_coverage,
                "system_acc": overall_system_acc,
                "expert_acc": overall_expert_acc,
                "classifier_acc": overall_classifier_acc,
                "alone_classifier": overall_alone_classifier_acc
            })
        print(to_print)
        if save_metrics_csv is not None:
            pd.DataFrame(save_metrics).to_csv(save_metrics_csv, index=False)

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
def run_reject(model, data_aug, n_dataset, expert_fn, epochs, alpha, T):
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

    kwargs = {'num_workers': 0, 'pin_memory': True}

    # Load the full CIFAR-10 dataset
    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                           transform=transform_train)
    
    # Calculate sizes based on the number of full sequences
    num_total_samples = len(train_dataset_all)
    num_total_sequences = num_total_samples // T
    
    train_seq_size = int(0.90 * num_total_sequences)
    test_seq_size = num_total_sequences - train_seq_size

    # Randomly split the dataset based on sequence chunks
    # This ensures that all images within a sequence stay together
    g = torch.Generator(device='cpu')
    g.manual_seed(42)
    indices = torch.randperm(num_total_sequences, generator=g)
    train_indices_seq = indices[:train_seq_size]
    test_indices_seq = indices[train_seq_size:]

    # Create the training and testing datasets with proper indices
    train_indices = []
    for seq_idx in train_indices_seq:
        train_indices.extend(range(seq_idx * T, (seq_idx + 1) * T))
    
    test_indices = []
    for seq_idx in test_indices_seq:
        test_indices.extend(range(seq_idx * T, (seq_idx + 1) * T))

    # Create subsets from the original dataset using the calculated indices
    train_subset = torch.utils.data.Subset(train_dataset_all, train_indices)
    test_subset = torch.utils.data.Subset(train_dataset_all, test_indices)

    # Wrap the subsets with IndexedCIFAR10
    train_dataset_indexed = IndexedCIFAR10(train_subset, T)
    test_dataset_indexed = IndexedCIFAR10(test_subset, T)

    # Create DataLoaders with shuffle=False to maintain sequential order
    train_loader = torch.utils.data.DataLoader(train_dataset_indexed,
                                               batch_size=128, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset_indexed,
                                               batch_size=128, shuffle=False, **kwargs)
    
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
            metrics_print(model, expert_fn, n_dataset, test_loader, T)
        metrics_print(model, expert_fn, n_dataset, test_loader, T, save_metrics_csv=os.path.join(result_root, f"general_model_curve_mixed_seed{seed}.csv"))

n_dataset = 10  # cifar-10

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


    @torch.no_grad()
    def predict(self, inputs, labels: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        device = labels.device
        B = labels.size(0)
        
        # Index expert accuracy by timestep t from t_tensor
        idx = t_tensor.long().cpu().numpy()

        p_master = self.acc_curve[idx]            # (B,)
        p_other  = self.other_acc_curve[idx]      # (B,)

        y_true = labels.detach().long().cpu().numpy()  # (B,)
        mask_master = (y_true < self.k)

        u = self.rng.random(B)
        hit = np.empty(B, dtype=bool)
        hit[mask_master] = (u[mask_master] < p_master[mask_master])
        hit[~mask_master] = (u[~mask_master] < p_other[~mask_master])

        def sample_any(y_np):
            any_class = self.rng.integers(0, self.n_classes, size=y_np.shape)
            return any_class

        preds = np.empty(B, dtype=np.int64)
        preds[hit]  = y_true[hit]
        preds[~hit] = sample_any(y_true[~hit])

        return torch.from_numpy(preds).to(device=device, dtype=torch.long)

import os, random, numpy as np, torch

run_seeds = [42, 43, 44, 45, 46]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

alpha = 1
epochs = 200
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_root = os.path.join(_CIFAR10_ROOT, "models", "general_curve_gaussian_k10_t50_no_linear", timestamp)
result_root = os.path.join(_CIFAR10_ROOT, "results", "general_curve_gaussian_k10_t50_no_linear", timestamp)
os.makedirs(save_root, exist_ok=True)
os.makedirs(result_root, exist_ok=True)

T = 50
# gap = len(p_curve) // T
# p_curve_new = []
# for i in range(T):
#     start = i * gap
#     end = min((i + 1) * gap, len(p_curve))
#     segment = p_curve[start:end]
#     avg = np.mean(segment)
#     p_curve_new.append(avg)

p_curve = np.array(p_curve)
# p_curve = np.linspace(1, 0.5, T)

p_low_curve = p_curve
# p_low_curve = np.linspace(1, 0.1, T)
# print(p_curve)

for seed in run_seeds:
    print(f"\n================ SEED {seed} ================\n")
    set_seed(seed)

    expert = SeqExpert(n_dataset, p_curve, p_low_curve, k=10, seed=seed)

    model = WideResNet(28, n_dataset + 1, 4, dropRate=0.0)

    run_reject(model, False, n_dataset, expert.predict, epochs, alpha, T)

    out_path = os.path.join(save_root, f"general_model_curve_mixed_seed{seed}.pth")
    torch.save(model, out_path)
    print(f"[seed={seed}] saved -> {out_path}")