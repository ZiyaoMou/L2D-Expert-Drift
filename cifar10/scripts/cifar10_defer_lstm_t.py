import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pandas as pd
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from common.utils import AverageMeter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from common.model import WideResNetRevised
from typing import Tuple, Any

_CIFAR10_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_default_expert = os.path.join(_CIFAR10_ROOT, "expert", "curve_new_85.csv")
exp_curve = pd.read_csv(os.environ.get("CIFAR10_EXPERT_CURVE", _default_expert))
p_curve = exp_curve['p']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class L2DLSTM(nn.Module):
    def __init__(self,
                 backbone,
                 hidden_dim: int,
                 num_layers: int,
                 n_classes: int,
                 dropout: float = 0.0,
                 max_T: int = 50,
                 normalize_t: bool = True,
                 cross_step_mode: str = "none"):
        super().__init__()
        assert cross_step_mode in ("none", "gru")  # kept for API compatibility with GRU variant
        self.backbone     = backbone
        self.hidden_dim   = int(hidden_dim)
        self.num_layers   = int(num_layers)
        self.n_classes    = int(n_classes)
        self.max_T        = int(max_T)
        self.normalize_t  = bool(normalize_t)
        self.cross_step_mode = cross_step_mode

        self.feat_dim = self.backbone.feat_dim
        self.fc_cls   = nn.Linear(self.feat_dim, self.n_classes)

        # Input matches GRU variant: concat [feat, h_prev, t_norm]
        self.rnn_input_dim = self.feat_dim + 2

        self.lstm = nn.LSTM(
            input_size  = self.rnn_input_dim,
            hidden_size = self.hidden_dim,
            num_layers  = self.num_layers,
            batch_first = True,      # expect [B, T, D]
            dropout     = (dropout if self.num_layers > 1 else 0.0),
            bidirectional=False
        )
        self.fc_def = nn.Linear(self.hidden_dim, 1)

    def _norm_t(self, t: torch.Tensor, dtype):
        if not self.normalize_t:
            return t.to(dtype)
        denom = max(1, self.max_T - 1)
        return (t.to(dtype) / denom)

    def forward(self,
                x: torch.Tensor,        # [B,T,C,H,W]
                h_prev: torch.Tensor,   # [B,T]
                t: torch.Tensor,        # [B,T]
                hidden_state=None) -> torch.Tensor:
        assert x.dim() == 5, f"x must be [B,T,C,H,W], got {x.shape}"
        B, T = x.shape[:2]
        assert h_prev.shape == (B, T) and t.shape == (B, T), \
            f"h_prev/t must be [B,T], got {h_prev.shape} / {t.shape}"

        x_flat    = x.reshape(B*T, *x.shape[2:])              # [B*T,C,H,W]
        feats_flat= self.backbone.forward_features(x_flat)    # [B*T,F]
        Fdim      = feats_flat.size(-1)
        feats     = feats_flat.view(B, T, Fdim)               # [B,T,F]

        logits_cls = self.fc_cls(feats)                       # [B,T,K]
        h_in  = h_prev.to(feats.dtype).unsqueeze(-1)          # [B,T,1]
        t_in  = self._norm_t(t, feats.dtype).unsqueeze(-1)    # [B,T,1]
        lstm_in = torch.cat([feats, h_in, t_in], dim=-1)      # [B,T,F+2]

        out, _ = self.lstm(lstm_in, hidden_state)             # out: [B,T,H]

        d_seq = self.fc_def(out)                               # [B,T,1]

        logits = torch.cat([logits_cls, d_seq], dim=-1)
        return logits


def train_reject_Tavg(train_loader, model, optimizer, scheduler, epoch,
                      expert_fn, n_classes, alpha, T):
    model.train()
    # if hasattr(model, "backbone"):
    #     model.backbone.eval()

    dev = next(model.parameters()).device
    batch_time = AverageMeter(); losses = AverageMeter(); top1 = AverageMeter()
    end = time.time()

    for i, (images, labels, seq_ids, ts) in enumerate(train_loader):
        images = images.to(dev, non_blocking=True).float()   # [B,C,H,W]
        labels = labels.to(dev, non_blocking=True).long()    # [B]
        ts     = ts.to(dev, non_blocking=True).long()        # [B]

        B = images.size(0)
        assert B % T == 0, f"batch_size({B}) must be divisible by T({T})"
        S = B // T

        x_seq  = images.view(S, T, *images.shape[1:])        # [S,T,C,H,W]
        y_seq  = labels.view(S, T)                           # [S,T]
        t_seq  = ts.view(S, T)                               # [S,T]

        with torch.no_grad():
            m_pred_flat = expert_fn(
                x_seq.reshape(S*T, *images.shape[1:]),       # [S*T,C,H,W]
                y_seq.reshape(-1),                           # [S*T]
                t_seq.reshape(-1)                            # [S*T]
            )                                                # [S*T]
        m_pred_seq = m_pred_flat.view(S, T)                  # [S,T]
        same_seq   = (m_pred_seq == y_seq).to(torch.float32) # [S,T] in {0,1}

        # h_prev[:,0]=0; remaining columns are same[t-1]
        h_prev_seq = torch.zeros(S, T, device=dev, dtype=same_seq.dtype)
        if T > 1:
            h_prev_seq[:, 1:] = same_seq[:, :-1]            # [S,T]

        # Single forward through L2DLSTM
        logits_seq = model(x_seq, h_prev_seq, t_seq)         # [S,T,K+1]

        # Deferral loss (vectorized)
        logp_seq   = F.log_softmax(logits_seq, dim=-1)       # [S,T,K+1]
        rc_idx     = torch.full((S, T, 1), n_classes, device=dev, dtype=torch.long)
        logp_defer = logp_seq.gather(-1, rc_idx).squeeze(-1)           # [S,T]
        logp_true  = logp_seq.gather(-1, y_seq.unsqueeze(-1)).squeeze(-1)  # [S,T]

        m   = same_seq
        m2  = torch.where(same_seq > 0.5,
                          torch.full_like(same_seq, float(alpha)),
                          torch.ones_like(same_seq))                     # [S,T]

        loss_t = -(m * logp_defer + m2 * logp_true)                      # [S,T]
        loss   = loss_t.mean()
        pred_seq   = logits_seq.argmax(dim=-1)                           # [S,T]
        is_def_seq = (pred_seq == n_classes)
        cls_ok     = ((~is_def_seq) & (pred_seq == y_seq)).sum().item()
        exp_ok     = (is_def_seq & (m_pred_seq == y_seq)).sum().item()
        correct_all = cls_ok + exp_ok
        total_all   = S * T
        acc_pct     = 100.0 * correct_all / total_all

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.update(float(loss.detach().cpu()), total_all)
        top1.update(acc_pct, total_all)
        batch_time.update(time.time() - end); end = time.time()

        if i % 10 == 0:
            print(f"[Deferral-Tavg][seq] Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Sys@1 {top1.val:.3f} ({top1.avg:.3f})")
    
    
@torch.no_grad()
def eval_reject_Tavg(val_loader, model, expert_fn, n_classes, alpha, T, desc="[VAL]"):
    model.eval()
    dev = next(model.parameters()).device
    loss_meter = AverageMeter()
    sys_correct = 0
    sys_total   = 0

    for images, labels, seq_ids, ts in val_loader:
        images = images.to(dev, non_blocking=True).float()
        labels = labels.to(dev, non_blocking=True).long()
        ts     = ts.to(dev, non_blocking=True).long()

        B = images.size(0)
        assert B % T == 0, f"batch_size({B}) must be divisible by T({T})"
        S = B // T

        x_seq  = images.view(S, T, *images.shape[1:])
        y_seq  = labels.view(S, T)
        t_seq  = ts.view(S, T)

        with torch.no_grad():
            m_pred_flat = expert_fn(
                x_seq.reshape(S*T, *images.shape[1:]),
                y_seq.reshape(-1),
                t_seq.reshape(-1)
            )
        m_pred_seq = m_pred_flat.view(S, T)
        same_seq   = (m_pred_seq == y_seq).to(torch.float32)

        h_prev_seq = torch.zeros(S, T, device=dev, dtype=same_seq.dtype)
        if T > 1:
            h_prev_seq[:, 1:] = same_seq[:, :-1]

        logits_seq = model(x_seq, h_prev_seq, t_seq)         # [S,T,K+1]

        logp_seq   = F.log_softmax(logits_seq, dim=-1)
        rc_idx     = torch.full((S, T, 1), n_classes, device=dev, dtype=torch.long)
        logp_defer = logp_seq.gather(-1, rc_idx).squeeze(-1)
        logp_true  = logp_seq.gather(-1, y_seq.unsqueeze(-1)).squeeze(-1)

        m   = same_seq
        m2  = torch.where(same_seq > 0.5,
                          torch.full_like(same_seq, float(alpha)),
                          torch.ones_like(same_seq))
        loss_t = -(m * logp_defer + m2 * logp_true)
        loss   = loss_t.mean().item()
        loss_meter.update(loss, S*T)

        pred_seq   = logits_seq.argmax(dim=-1)
        is_def_seq = (pred_seq == n_classes)
        cls_ok     = ((~is_def_seq) & (pred_seq == y_seq)).sum().item()
        exp_ok     = (is_def_seq & (m_pred_seq == y_seq)).sum().item()
        sys_correct += (cls_ok + exp_ok)
        sys_total   += S*T

    sys_acc = 100.0 * sys_correct / max(1, sys_total)
    print(f"{desc}  | Avg Loss: {loss_meter.avg:.4f} | Sys@1: {sys_acc:.3f}% | N={sys_total}")
    return {"val_loss": loss_meter.avg, "sys_acc": sys_acc, "n": sys_total}


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


@torch.no_grad()
def metrics_print_seq(net, expert_fn, n_classes, loader, T, save_metrics_csv=None):
    net.eval()
    if hasattr(net, "backbone") and hasattr(net.backbone, "eval"):
        net.backbone.eval()

    dev = next(net.parameters()).device

    step_metrics = [{
        "correct_sys": 0, "exp": 0, "correct_cls": 0, "alone_correct": 0,
        "real_total": 0, "exp_total": 0, "cls_total": 0
    } for _ in range(T)]
    overall = {k: 0 for k in ["correct_sys","exp","correct_cls","alone_correct",
                              "real_total","exp_total","cls_total"]}

    for images, labels, seq_ids, ts in loader:
        images = images.to(dev, non_blocking=True).float()
        labels = labels.to(dev, non_blocking=True).long()
        ts     = ts.to(dev, non_blocking=True).long()

        B = images.size(0)
        assert B % T == 0, f"batch_size({B}) must be divisible by T({T})"
        S = B // T

        x_seq  = images.view(S, T, *images.shape[1:])
        y_seq  = labels.view(S, T)
        t_seq  = ts.view(S, T)

        m_pred_flat = expert_fn(
            x_seq.reshape(S*T, *images.shape[1:]),
            y_seq.reshape(-1),
            t_seq.reshape(-1)
        )
        m_pred_seq = m_pred_flat.view(S, T)
        is_exp_ok  = (m_pred_seq == y_seq)                 # [S,T], bool

        h_prev_seq = torch.zeros(S, T, device=dev, dtype=torch.float32)
        if T > 1:
            h_prev_seq[:, 1:] = is_exp_ok[:, :-1].to(torch.float32)

        logits_seq = net(x_seq, h_prev_seq, t_seq)         # [S,T,K+1]
        pred_seq   = logits_seq.argmax(dim=-1)             # [S,T]
        alone_pred = logits_seq[..., :n_classes].argmax(dim=-1)  # [S,T]
        is_def_seq = (pred_seq == n_classes)

        for t in range(T):
            m = step_metrics[t]
            alone_ok = (alone_pred[:, t] == y_seq[:, t]).sum().item()
            m["alone_correct"] += alone_ok
            overall["alone_correct"] += alone_ok

            cls_mask = ~is_def_seq[:, t]
            exp_mask =  is_def_seq[:, t]

            cls_ok = (pred_seq[:, t][cls_mask] == y_seq[:, t][cls_mask]).sum().item()
            exp_ok = (is_exp_ok[:, t][exp_mask]).sum().item()

            m["correct_cls"] += cls_ok
            m["cls_total"]   += int(cls_mask.sum().item())
            m["exp"]         += exp_ok
            m["exp_total"]   += int(exp_mask.sum().item())
            m["correct_sys"] += (cls_ok + exp_ok)
            m["real_total"]  += x_seq.size(0)

            overall["correct_cls"] += cls_ok
            overall["cls_total"]   += int(cls_mask.sum().item())
            overall["exp"]         += exp_ok
            overall["exp_total"]   += int(exp_mask.sum().item())
            overall["correct_sys"] += (cls_ok + exp_ok)
            overall["real_total"]  += x_seq.size(0)

    rows = []
    print("\n--- Per-Time-Step Metrics ---")
    for t in range(T):
        m = step_metrics[t]
        if m["real_total"] == 0: continue
        coverage       = m["cls_total"] / m["real_total"]
        system_acc     = 100.0 * m["correct_sys"] / m["real_total"]
        expert_acc     = 100.0 * m["exp"] / (m["exp_total"] + 1e-5)
        classifier_acc = 100.0 * m["correct_cls"] / (m["cls_total"] + 1e-5)
        alone_cls_acc  = 100.0 * m["alone_correct"] / m["real_total"]
        if save_metrics_csv is not None:
            rows.append({
                "time_step": t+1,
                "coverage": coverage,
                "system_acc": system_acc,
                "expert_acc": expert_acc,
                "classifier_acc": classifier_acc,
                "alone_classifier": alone_cls_acc
            })

    print("\n--- Overall Metrics (Average over all time steps) ---")
    if overall["real_total"] > 0:
        overall_coverage       = overall["cls_total"] / overall["real_total"]
        overall_system_acc     = 100.0 * overall["correct_sys"] / overall["real_total"]
        overall_expert_acc     = 100.0 * overall["exp"] / (overall["exp_total"] + 1e-5)
        overall_classifier_acc = 100.0 * overall["correct_cls"] / (overall["cls_total"] + 1e-5)
        overall_alone_acc      = 100.0 * overall["alone_correct"] / overall["real_total"]

        to_print = {
            "coverage": f"{overall['cls_total']} out of {overall['real_total']}",
            "system accuracy": overall_system_acc,
            "expert accuracy": overall_expert_acc,
            "classifier accuracy": overall_classifier_acc,
            "alone classifier": overall_alone_acc
        }
        print(to_print)

        if save_metrics_csv is not None:
            rows.append({
                "time_step": "overall",
                "coverage": overall_coverage,
                "system_acc": overall_system_acc,
                "expert_acc": overall_expert_acc,
                "classifier_acc": overall_classifier_acc,
                "alone_classifier": overall_alone_acc
            })
            pd.DataFrame(rows).to_csv(save_metrics_csv, index=False)
        return overall_system_acc

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

    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                           transform=transform_train)
    subset_size = int(1 * len(train_dataset_all))
    remaining_size = len(train_dataset_all) - subset_size
    train_dataset_small, _ = torch.utils.data.random_split(
        train_dataset_all, [subset_size, remaining_size],
        generator=torch.Generator().manual_seed(42)
    )
        
    num_total_samples = len(train_dataset_small)
    num_total_sequences = num_total_samples // T
    
    train_seq_size = int(0.90 * num_total_sequences)
    test_seq_size = num_total_sequences - train_seq_size

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset_indexed,
        batch_size=500, shuffle=False, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset_indexed,
        batch_size=500, shuffle=False, **kwargs
    )
        
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print(len(train_loader))

    model = model.to(device)

    cudnn.benchmark = True

    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)
    best_epoch = -1
    best_path = os.path.join(save_root, f"best_deferral_{seed}.pth")
    sys_acc = metrics_print_seq(model, expert_fn, n_dataset, test_loader, T)
    best_acc = sys_acc
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_reject_Tavg(train_loader, model, optimizer, scheduler, epoch,
                        expert_fn, n_dataset, alpha, T)
        sys_acc = metrics_print_seq(model, expert_fn, n_dataset, test_loader, T)
        eval_ = eval_reject_Tavg(test_loader, model, expert_fn, n_dataset, alpha, T)
        val_loss = eval_["val_loss"]
        best_acc = sys_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] epoch={epoch} best_val_loss={best_val_loss:.4f} best_acc={best_acc:.4f} -> saved to {best_path}")

        # if sys_acc > best_acc:
        #     best_acc = sys_acc
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), best_path)
        #     print(f"[BEST] epoch={epoch} best_acc={best_acc:.4f} -> saved to {best_path}")

    print(f"Training done. Best epoch={best_epoch}, best acc={best_acc:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=next(model.parameters()).device))
    metrics_print_seq(model, expert_fn, n_dataset, test_loader, T,
                        save_metrics_csv=os.path.join(result_root, f"general_model_curve_mixed_seed{seed}.csv"))


alpha = 1
epochs = 200
n_dataset = 10
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_root = os.path.join(_CIFAR10_ROOT, "models", "lstm_curve_sort_85", timestamp)
result_root = os.path.join(_CIFAR10_ROOT, "results", "lstm_curve_sort_85", timestamp)
os.makedirs(save_root, exist_ok=True)
os.makedirs(result_root, exist_ok=True)

T = 50
gap = len(p_curve) // T
p_curve_new = []
for i in range(T):
    start = i * gap
    end = min((i + 1) * gap, len(p_curve))
    segment = p_curve[start:end]
    avg = np.mean(segment)
    p_curve_new.append(avg)

p_curve = np.array(p_curve_new)
p_low_curve = np.full(T, 0.1)
# p_low_curve = np.linspace(1, 0.1, T)
print(p_curve)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

run_seeds = [42, 43, 44, 45, 46]

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


for seed in run_seeds:
    print(f"\n================ SEED {seed} ================\n")
    set_seed(seed)

    expert = SeqExpert(n_dataset, p_curve, p_low_curve, k=7, seed=seed)

    backbone = WideResNetRevised(28, n_dataset, 4, dropRate=0.0)

    ckpt_path = os.environ.get(
        "CIFAR10_BACKBONE_CKPT",
        os.path.join(_CIFAR10_ROOT, "models", "model_cls_backbone_weight.pth"),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone_state = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(backbone_state)
    
    model = L2DLSTM(backbone, hidden_dim=512, num_layers=1, n_classes=n_dataset, dropout=0.2)

    run_reject(model, False, n_dataset, expert.predict, epochs, alpha, T)

    out_path = os.path.join(save_root, f"general_model_curve_mixed_seed{seed}.pth")
    torch.save(model, out_path)
    print(f"[seed={seed}] saved -> {out_path}")