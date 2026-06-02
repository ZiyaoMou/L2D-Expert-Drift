import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import torchvision
from experts.fake_bias import ExpertModelBiased
from utils.loss import compute_AUROC
import pandas as pd
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score  


# alpha = torch.tensor([1,1,0.1,1,1,0.4,0.2,1,0.1,1,0.1,1,1,1])
alpha = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1])

def CrossEntropyLoss_defer(outputs, labels, expert_preds, weights, alpha):
    """
    Soft Learn-to-Defer loss, vectorized.

    Args:
      outputs:       [B, K, 3] — per-class logits: [logit_class0, logit_class1, logit_defer]
      labels:        [B, K]    — ground-truth {0,1}
      expert_preds:  [B, K]    — expert predictions {0,1}
      weights:       [B, K]    — sample weights
      alpha:         [K]       — per-class α

    Returns:
      scalar loss
    """
    B, K, _ = outputs.shape
    device = outputs.device
    p = F.softmax(outputs, dim=2)
    p_true   = p.gather(2, labels.long().unsqueeze(2)).squeeze(2)
    p_reject = p[..., 2]
    expert_correct = (expert_preds == labels).float()
    expert_wrong   = 1.0 - expert_correct
    alpha = alpha.to(device).view(1, K)
    w_cls = alpha * expert_correct + expert_wrong

    loss_cls = - w_cls * weights * torch.log2(p_true + 1e-12)
    loss_def = - weights * expert_correct * torch.log2(p_reject + 1e-12)

    denom = weights.sum(dim=0).clamp(min=1e-12)
    cls_per_class = loss_cls.sum(dim=0) / denom
    def_per_class = loss_def.sum(dim=0) / denom

    return (cls_per_class + def_per_class).mean()


def CrossEntropyLoss_no_defer(outputs, labels, weights):
    """
    Standard cross-entropy (no defer).

    Args:
      outputs:      [B, K, 2] — per-class logits for the two labels
      labels:       [B, K]    — ground-truth {0,1}
      expert_preds: unused
      weights:      [B, K]    — sample weights
      alpha:        unused

    Returns:
      scalar loss
    """
    p = F.softmax(outputs, dim=2)
    p_true = p.gather(2, labels.long().unsqueeze(2)).squeeze(2)

    loss = - weights * torch.log2(p_true + 1e-12)
    denom = weights.sum(dim=0).clamp(min=1e-12)
    per_class = loss.sum(dim=0) / denom
    return per_class.mean()


class DenseNet121SeqDefer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.densenet121(pretrained=True)
        feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(feat_dim, num_classes * 3)
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

class SeqTrainerDefer:
    def __init__(self, model, expert: ExpertModelBiased, device, seed=None,
                 result_dir="results/general_curve_21steps"):
        self.model = model.to(device)
        self.expert = expert
        self.device = device
        self.seed = seed
        self.result_dir = result_dir

    def train_epoch(self, loader, optimizer, use_defer=True):
        self.model.train()
        running_loss = 0.
        for imgs_seq, labels_seq, weights_seq, rad_1, rad_2, rad_3 in tqdm(loader, desc="Train"):
            B, T, K = labels_seq.shape
            imgs_seq = imgs_seq.to(self.device)
            labels_seq = labels_seq.to(self.device)
            weights_seq = weights_seq.to(self.device)
            rad_1 = rad_1.to(self.device)
            rad_2 = rad_2.to(self.device)
            rad_3 = rad_3.to(self.device)

            loss_sum = 0.
            for t in range(T):
                x_t = imgs_seq[:, t]
                B_, C, H, W = x_t.shape
                x_flat = x_t.view(B_, C, H, W)
                out = self.model(x_flat)
                g = out.view(B_, self.model.num_classes, 3)
                g_cls = g[..., :2]

                if use_defer:
                    loss_t = CrossEntropyLoss_defer(g, labels_seq[:, t], rad_1[:, t], weights_seq[:, t], alpha)
                else:
                    loss_t = CrossEntropyLoss_no_defer(g_cls, labels_seq[:, t], weights_seq[:, t])
                loss_sum += loss_t

            loss = loss_sum / T
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)

    def validate_epoch(self, loader, use_defer=True, epoch_num=0, save_results=False):
        self.model.eval()
        running_loss = 0.0
        n_batch = 0

        all_labels = []
        all_pcls = []
        all_pexpr = []
        all_psys = []
        all_defer = []
        all_weights = []
        total_defer = 0
        total_preds = 0

        with torch.no_grad():
            progress_bar = tqdm(loader, leave=False, ncols=100)
            for batch_idx, (imgs, lbs, wts, rad_1, rad_2, rad_3) in enumerate(progress_bar):
                imgs = imgs.to(self.device)
                lbs = lbs.to(self.device)
                wts = wts.to(self.device)
                rad_1 = rad_1.to(self.device)
                rad_2 = rad_2.to(self.device)
                rad_3 = rad_3.to(self.device)
                B, T, K = lbs.shape
                is_pretraining = not use_defer
                exp = rad_1

                loss_sum = 0.0
                batch_labels = []
                batch_pcls = []
                batch_pexpr = []
                batch_psys = []
                batch_defer = []
                batch_weights = []

                for t in range(T):
                    backbone_out = self.model.backbone(imgs[:, t])
                    backbone_out = backbone_out.view(B, K, 3)
                    g_cls = backbone_out[:, :, :2]
                    g_def = backbone_out[:, :, 2]
                    g_def_expanded = g_def.unsqueeze(-1)
                    g_out = torch.cat([g_cls, g_def_expanded], dim=2)

                    if is_pretraining:
                        loss_sum += CrossEntropyLoss_no_defer(g_cls, lbs[:, t], wts[:, t])
                        sys_pred = F.softmax(g_cls, dim=2)[:, :, 1]
                        batch_pexpr.append(np.zeros((B, K), dtype=np.float32))
                        batch_defer.append(np.zeros((B, K), dtype=np.bool_))
                    else:
                        loss_sum += CrossEntropyLoss_defer(g_out, lbs[:, t], exp[:, t], wts[:, t], alpha)
                        p = F.softmax(g_out, dim=2)
                        p_class0 = p[:, :, 0]
                        p_class1 = p[:, :, 1]
                        p_defer = p[:, :, 2]
                        max_cls_prob = torch.max(p_class0, p_class1)
                        mask = p_defer > max_cls_prob
                        sys_pred = torch.where(mask, exp[:, t], p_class1)
                        batch_pexpr.append(exp[:, t].cpu().numpy())
                        batch_defer.append(mask.cpu().numpy())
                        
                        total_defer += mask.sum().item()
                        total_preds += mask.numel()

                    batch_labels.append(lbs[:, t].cpu().numpy())
                    batch_pcls.append(F.softmax(g_cls, dim=2)[:, :, 1].cpu().numpy())
                    batch_psys.append(sys_pred.cpu().numpy())
                    batch_weights.append(wts[:, t].cpu().numpy())

                loss = loss_sum / T
                running_loss += loss.item()
                n_batch += 1
                progress_bar.set_postfix(loss=f"Val Loss: {loss:.4f}")

                all_labels.append(np.stack(batch_labels, axis=0))
                all_pcls.append(np.stack(batch_pcls, axis=0))
                all_pexpr.append(np.stack(batch_pexpr, axis=0))
                all_psys.append(np.stack(batch_psys, axis=0))
                all_defer.append(np.stack(batch_defer, axis=0))
                all_weights.append(np.stack(batch_weights, axis=0))

        all_labels = np.concatenate(all_labels, axis=1)
        all_pcls = np.concatenate(all_pcls, axis=1)
        all_pexpr = np.concatenate(all_pexpr, axis=1)
        all_psys = np.concatenate(all_psys, axis=1)
        all_defer = np.concatenate(all_defer, axis=1)
        all_weights = np.concatenate(all_weights, axis=1)

        K = self.model.num_classes

        T = all_labels.shape[0]
        auc_cls_per_class_timestep = np.full((K, T), np.nan)
        auc_exp_per_class_timestep = np.full((K, T), np.nan)
        auc_sys_per_class_timestep = np.full((K, T), np.nan)
        auprc_cls_per_class_timestep = np.full((K, T), np.nan)
        auprc_exp_per_class_timestep = np.full((K, T), np.nan)
        auprc_sys_per_class_timestep = np.full((K, T), np.nan)
        defer_rates_per_class_timestep = np.full((K, T), np.nan)
        
        for t in range(T):
            for i in range(K):
                mask_t_i = all_weights[t, :, i] == 1
                if mask_t_i.sum() > 1 and len(np.unique(all_labels[t, mask_t_i, i])) > 1:
                    auc_cls_per_class_timestep[i, t] = roc_auc_score(all_labels[t, mask_t_i, i], all_pcls[t, mask_t_i, i])
                    auc_exp_per_class_timestep[i, t] = roc_auc_score(all_labels[t, mask_t_i, i], all_pexpr[t, mask_t_i, i])
                    auc_sys_per_class_timestep[i, t] = roc_auc_score(all_labels[t, mask_t_i, i], all_psys[t, mask_t_i, i])
                    auprc_cls_per_class_timestep[i, t] = average_precision_score(all_labels[t, mask_t_i, i], all_pcls[t, mask_t_i, i])
                    auprc_exp_per_class_timestep[i, t] = average_precision_score(all_labels[t, mask_t_i, i], all_pexpr[t, mask_t_i, i])
                    auprc_sys_per_class_timestep[i, t] = average_precision_score(all_labels[t, mask_t_i, i], all_psys[t, mask_t_i, i])
                defer_rates_per_class_timestep[i, t] = all_defer[t, :, i].mean() if not is_pretraining else 0.0
        
        auc_cls_per_class = np.nanmean(auc_cls_per_class_timestep, axis=1)
        auc_exp_per_class = np.nanmean(auc_exp_per_class_timestep, axis=1)
        auc_sys_per_class = np.nanmean(auc_sys_per_class_timestep, axis=1)
        auprc_cls_per_class = np.nanmean(auprc_cls_per_class_timestep, axis=1)
        auprc_exp_per_class = np.nanmean(auprc_exp_per_class_timestep, axis=1)
        auprc_sys_per_class = np.nanmean(auprc_sys_per_class_timestep, axis=1)
        defer_rates = np.nanmean(defer_rates_per_class_timestep, axis=1)

        auc_cls = np.nanmean(auc_cls_per_class)
        auc_exp = np.nanmean(auc_exp_per_class)
        auc_sys = np.nanmean(auc_sys_per_class)
        auprc_cls = np.nanmean(auprc_cls_per_class)
        auprc_exp = np.nanmean(auprc_exp_per_class)
        auprc_sys = np.nanmean(auprc_sys_per_class)
        defer_rate = defer_rates.mean()

        if not is_pretraining:
            defer_rate = total_defer / total_preds
            print(f"[Val] AUC_cls={auc_cls:.4f}, AUC_exp={auc_exp:.4f}, "
                f"AUC_sys={auc_sys:.4f}, Defer Rate={defer_rate:.4f}")
            print(f"[Val] AUPRC_cls={auprc_cls:.4f}, AUPRC_exp={auprc_exp:.4f}, AUPRC_sys={auprc_sys:.4f}")
        else:
            defer_rate = 0.0
            print(f"[Val] AUC_cls={auc_cls:.4f}, AUC_exp={auc_exp:.4f}, AUC_sys={auc_sys:.4f}")
            print(f"[Val] AUPRC_cls={auprc_cls:.4f}, AUPRC_exp={auprc_exp:.4f}, AUPRC_sys={auprc_sys:.4f}")

        if save_results:
            df = pd.DataFrame()
            for i in range(K):
                for t in range(T):
                    new_row = pd.DataFrame({
                        'Class': i,
                        'Timestep': t,
                        'AUC_cls': auc_cls_per_class_timestep[i, t],
                        'AUC_exp': auc_exp_per_class_timestep[i, t],
                        'AUC_sys': auc_sys_per_class_timestep[i, t],
                        'AUPRC_cls': auprc_cls_per_class_timestep[i, t],
                        'AUPRC_exp': auprc_exp_per_class_timestep[i, t],
                        'AUPRC_sys': auprc_sys_per_class_timestep[i, t],
                        'Defer_Rate': defer_rates_per_class_timestep[i, t]
                    }, index=[0])
                    df = pd.concat([df, new_row], ignore_index=True)


            out_dir = f"{self.result_dir}/{self.seed}"
            os.makedirs(out_dir, exist_ok=True)
            detailed_path = f"{out_dir}/seq_eval_detailed-epoch-{epoch_num}-no-alpha-{T}-steps.csv"
            df.to_csv(detailed_path, index=False)

            timestep_summary = pd.DataFrame()
            for t in range(T):
                new_row = pd.DataFrame({
                    'Timestep': t,
                    'AUC_cls_mean': np.nanmean(auc_cls_per_class_timestep[:, t]),
                    'AUC_exp_mean': np.nanmean(auc_exp_per_class_timestep[:, t]),
                    'AUC_sys_mean': np.nanmean(auc_sys_per_class_timestep[:, t]),
                    'AUPRC_cls_mean': np.nanmean(auprc_cls_per_class_timestep[:, t]),
                    'AUPRC_exp_mean': np.nanmean(auprc_exp_per_class_timestep[:, t]),
                    'AUPRC_sys_mean': np.nanmean(auprc_sys_per_class_timestep[:, t]),
                    'Defer_Rate_mean': np.nanmean(defer_rates_per_class_timestep[:, t])
                }, index=[0])
                timestep_summary = pd.concat([timestep_summary, new_row], ignore_index=True)

            timestep_summary.to_csv(
                f"{out_dir}/seq_eval_timestep_summary-epoch-{epoch_num}-no-alpha-{T}-steps.csv",
                index=False
            )

        cls_aurocs = [np.nanmean(auc_cls_per_class_timestep[:, t]) for t in range(T)]
        aurocs_exp = [np.nanmean(auc_exp_per_class_timestep[:, t]) for t in range(T)]
        system_aurocs = [np.nanmean(auc_sys_per_class_timestep[:, t]) for t in range(T)]
        defer_rates_timestep = [np.nanmean(defer_rates_per_class_timestep[:, t]) for t in range(T)]

        return running_loss / n_batch, cls_aurocs, aurocs_exp, system_aurocs, defer_rates_timestep

    def fit(self, train_loader, val_loader, pretained_epochs, finetuned_epochs, lr=1e-4,
            pretrain_ckpt_dir=None):
        self.pretrained_epochs = pretained_epochs
        self.finetuned_epochs = finetuned_epochs
        
        optimizer = optim.Adam (self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, mode='min')

        for e in range(pretained_epochs):
            tr_loss = self.train_epoch(train_loader, optimizer, use_defer=False)
            val_loss, cls_aurocs, exp_aurocs, sys_aurocs, _ = self.validate_epoch(val_loader, use_defer=False, epoch_num=e, save_results=True)
            # self.validate_epoch_best_coverage(val_loader, rad_index=1)
            
            scheduler.step(val_loss)
            print(f"Epoch {e+1}/{pretained_epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"AUC_cls={np.nanmean(cls_aurocs):.4f}  AUC_exp={np.nanmean(exp_aurocs):.4f}  AUC_sys={np.nanmean(sys_aurocs):.4f}")
            if pretrain_ckpt_dir:
                os.makedirs(pretrain_ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(pretrain_ckpt_dir, f"pretrain_epoch_{e+1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"[CKPT] Saved pretrained (no-defer) state -> {ckpt_path}")

        for e in range(finetuned_epochs):
            tr_loss = self.train_epoch(train_loader, optimizer, use_defer=True)
            val_loss, cls_aurocs, exp_aurocs, sys_aurocs, defer_rates = self.validate_epoch(val_loader, use_defer=True, epoch_num=e, save_results=True)

            print(f"Finetune {e+1}/{finetuned_epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"AUC_cls={np.nanmean(cls_aurocs):.4f}  AUC_exp={np.nanmean(exp_aurocs):.4f}  AUC_sys={np.nanmean(sys_aurocs):.4f}  "
                  f"Defer_Rate={np.nanmean(defer_rates):.4f}")
        
        return self.model