import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv, os
from sklearn.metrics import roc_auc_score, average_precision_score
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
from utils.loss import compute_AUROC
import pandas as pd

# alpha: per-pathology weights; default is ones on the same device as outputs
def l2d_loss(outputs, labels, expert_preds, weights, alpha=None):
    """
    Soft Learn-to-Defer loss, vectorized.

    Args:
      outputs:       [B, K, 3] — per-class logits: [logit_class0, logit_class1, logit_defer]
      labels:        [B, K]    — ground-truth {0,1}
      expert_preds:  [B, K]    — expert predictions {0,1}
      weights:       [B, K]    — sample weights
      alpha:         [K]       — per-class α (optional; defaults to ones)

    Returns:
      scalar loss
    """
    B, K, _ = outputs.shape
    device = outputs.device
    dtype = outputs.dtype
    if alpha is None:
        alpha = torch.ones(K, device=device, dtype=dtype)
    else:
        alpha = alpha.to(device=device, dtype=dtype).reshape(-1)
        if alpha.numel() == 1:
            alpha = alpha.expand(K)
        elif alpha.numel() != K:
            raise ValueError(f"alpha must have length K={K} or be scalar, got shape {alpha.shape}")
    p = F.softmax(outputs, dim=2)
    p_true   = p.gather(2, labels.long().unsqueeze(2)).squeeze(2)
    p_reject = p[..., 2]
    expert_correct = (expert_preds == labels).float()
    expert_wrong   = 1.0 - expert_correct
    alpha = alpha.view(1, K)
    w_cls = alpha * expert_correct + expert_wrong

    loss_cls = - w_cls * weights * torch.log2(p_true + 1e-12)
    loss_def = - weights * expert_correct * torch.log2(p_reject + 1e-12)

    denom = weights.sum(dim=0).clamp(min=1e-12)
    cls_per_class = loss_cls.sum(dim=0) / denom
    def_per_class = loss_def.sum(dim=0) / denom

    return (cls_per_class + def_per_class).mean()


def l2d_loss_no_defer(g_cls, labels, weights):
    # """
    # g_cls   : [B, K, 2] — classifier logits per class
    # g_def   : [B, K]     — unused
    # labels  : [B, K]     — binary ground truth labels
    # weights : [B, K]     — optional sample weights
    # alpha   : ignored
    # """
    B, K = labels.shape
    p_cls = F.softmax(g_cls, dim=2)
    total_loss = 0.0
    for i in range(K):
        batch_indices = torch.arange(B, device=labels.device)
        p_class_i = p_cls[:, i, :][batch_indices, labels[:, i].long()]
        loss_clf = -weights[:, i] * torch.log(p_class_i + 1e-12)
        loss_total_i = loss_clf.sum() / (weights[:, i].sum() + 1e-8)
        total_loss += loss_total_i
    total_loss = total_loss / K
    return total_loss


class DenseLSTMDefer(nn.Module):
    def __init__(self, num_classes, lstm_hidden=512, lstm_layers=1):
        super().__init__()
        backbone = torchvision.models.densenet121(pretrained=True)
        self.cnn = nn.Sequential(
            backbone.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.num_classes = num_classes
        self.feat_dim = backbone.classifier.in_features
        self.clf_head = nn.Linear(self.feat_dim, num_classes * 2)
        self.lstm = nn.LSTM(self.feat_dim + num_classes, lstm_hidden,
                                lstm_layers, batch_first=True)
        self.def_head = nn.Linear(lstm_hidden, num_classes)
        self.hidden_dim = lstm_hidden
        self.lstm_layers = lstm_layers

    def forward(self, x_seq, h_prev_s=None, pretraining=False):
        B, T, C, H, W = x_seq.shape
        x = x_seq.view(B*T, C, H, W)
        f = self.cnn(x).flatten(1)
        f = f.view(B, T, -1)
        g_cls = self.clf_head(f)
        lstm_in = torch.cat([f, h_prev_s], dim=2)
        lstm_out, _ = self.lstm(lstm_in)
        g_def = self.def_head(lstm_out)
        g_all = torch.cat([g_cls.view(B, self.num_classes, 2), g_def.unsqueeze(-1)], dim=2).view(B, -1)
        return g_all

class CheXpertTrainerDeferLSTM:
    def __init__(self, model, expert, device, lr=0.0001, pretrained_epochs=3, finetuned_epochs=10,
                 seq_len=10, step=10, result_dir="results/lstm_curve_30032026",
                 checkpoint_dir=None, epoch_offset=0):
        self.model = model
        self.expert = expert
        self.device = device
        self.lr = lr
        self.pretrained_epochs = pretrained_epochs
        self.finetuned_epochs = finetuned_epochs
        self.seq_len = seq_len
        self.step = step
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir   # if set, save best-val checkpoint here
        self.epoch_offset = epoch_offset       # for labelling epochs when resuming

    def train_defer_lstm(self, train_loader, test_loader, seed):
        self.model.train()

        # Pretrain phase — skipped when pretrained_epochs=0 (shared checkpoint
        # already loaded by caller).
        if self.pretrained_epochs > 0:
            print(f"[LSTM] Starting {self.pretrained_epochs} pretrained epoch(s)...")
            for epoch in range(self.pretrained_epochs):
                optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
                scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, mode='min')
                train_loss = self.train_epoch(train_loader, optimizer, l2d_loss_no_defer)
                test_results = self.test_epoch(test_loader, l2d_loss_no_defer, epoch, seed)
                test_loss, auc_scores_test, auc_scores_exp_test, auc_scores_sys_test, defer_rates, auprc_scores_test, auprc_scores_exp_test, auprc_scores_sys_test = test_results[:8]
                test_auroc = np.nanmean(auc_scores_test)
                test_auroc_exp = np.nanmean(auc_scores_exp_test)
                test_auroc_sys = np.nanmean(auc_scores_sys_test)
                scheduler.step(train_loss)
                print(f"Pretrained Epoch {epoch+1}/{self.pretrained_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auroc:.4f}, Test AUC Exp: {test_auroc_exp:.4f}, Test AUC Sys: {test_auroc_sys:.4f}")
        else:
            print(f"[LSTM] pretrained_epochs=0, skipping pretrain (shared checkpoint already loaded).")

        finetune_train_losses = []
        finetune_test_losses  = []
        best_val_loss  = float('inf')
        best_epoch_idx = -1

        os.makedirs(self.result_dir, exist_ok=True)
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(self.finetuned_epochs):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
            scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, mode='min')
            train_loss = self.train_epoch(train_loader, optimizer, l2d_loss)
            abs_epoch  = self.epoch_offset + epoch   # absolute epoch index for CSV naming
            test_results = self.test_epoch(test_loader, l2d_loss, abs_epoch, seed)
            test_loss, auc_scores_test, auc_scores_exp_test, auc_scores_sys_test, defer_rates, auprc_scores_test, auprc_scores_exp_test, auprc_scores_sys_test = test_results[:8]
            finetune_train_losses.append(float(train_loss))
            finetune_test_losses.append(float(test_loss))

            test_auroc     = np.nanmean(auc_scores_test)
            test_auroc_sys = np.nanmean(auc_scores_sys_test)
            scheduler.step(train_loss)

            # Save best-val checkpoint
            if self.checkpoint_dir and test_loss < best_val_loss:
                best_val_loss  = test_loss
                best_epoch_idx = abs_epoch
                ckpt_path = os.path.join(self.checkpoint_dir, f"best_val_seed{seed}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [best-val] epoch {abs_epoch+1}, val_loss={test_loss:.5f} -> saved {ckpt_path}")

            print(f"Finetune Epoch {abs_epoch+1} (local {epoch+1}/{self.finetuned_epochs}), "
                  f"Train={train_loss:.4f}, Val={test_loss:.4f}, "
                  f"AUC_sys={test_auroc_sys:.4f}")

        # Append losses to CSV (supports resuming)
        csv_path = f'{self.result_dir}/finetune_losses_seed{seed}.csv'
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'test_loss'])
            start_ep = self.epoch_offset + 1
            for i, (tr, te) in enumerate(zip(finetune_train_losses, finetune_test_losses)):
                writer.writerow([start_ep + i, tr, te])
        print(f'[INFO] Losses appended to {csv_path}')
        if self.checkpoint_dir and best_epoch_idx >= 0:
            print(f'[INFO] Best val epoch={best_epoch_idx+1}, loss={best_val_loss:.5f}')
    def train_epoch(self, train_loader, optimizer, loss_fn):
        self.model.train()
        running_loss = 0.0
        n_batch = 0
        progress_bar = tqdm(train_loader, leave=False, ncols=100)

        for imgs, lbs, wts, rad_1, rad_2, rad_3 in progress_bar:
            imgs = imgs.to(self.device)
            lbs = lbs.to(self.device)
            wts = wts.to(self.device)
            rad_1 = rad_1.to(self.device)
            rad_2 = rad_2.to(self.device)
            rad_3 = rad_3.to(self.device)
            B, T, K = lbs.shape
            is_pretraining = loss_fn == l2d_loss_no_defer

            hidden = None
            loss_sum = 0.0

            exp = rad_1

            for t in range(T):
                f_t = self.model.cnn(imgs[:, t]).flatten(1)
                g_cls = self.model.clf_head(f_t).view(B, K, 2)
                if t == 0:
                    exp_prev = torch.zeros(B, K, device=self.device)
                    lstm_in = torch.cat([f_t, exp_prev], 1).unsqueeze(1)
                else:
                    lstm_in = torch.cat([f_t, exp[:, t-1]], 1).unsqueeze(1)
                lstm_out, hidden = self.model.lstm(lstm_in, hidden)
                g_def = self.model.def_head(lstm_out.squeeze(1))  
                g_def_expanded = g_def.unsqueeze(-1)
                g_out = torch.cat([g_cls, g_def_expanded], dim=2)                   

                if is_pretraining:
                    loss_sum += loss_fn(g_cls, lbs[:, t], wts[:, t])
                else:
                    loss_sum += loss_fn(g_out, lbs[:, t], exp[:, t], wts[:, t])
                    
            loss = loss_sum / T
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batch += 1
            avg_loss = running_loss / n_batch
            progress_bar.set_postfix(loss=f"Train Loss: {avg_loss:.4f}")

        return running_loss / n_batch

    def test_epoch(self, test_loader, loss_fn, epoch_num, seed):
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
            progress_bar = tqdm(test_loader, leave=False, ncols=100)
            for batch_idx, (imgs, lbs, wts, rad_1, rad_2, rad_3) in enumerate(progress_bar):
                imgs = imgs.to(self.device)
                lbs = lbs.to(self.device)
                wts = wts.to(self.device)
                rad_1 = rad_1.to(self.device)
                rad_2 = rad_2.to(self.device)
                rad_3 = rad_3.to(self.device)
                B, T, K = lbs.shape
                is_pretraining = loss_fn == l2d_loss_no_defer
                exp = rad_1

                hidden = None
                loss_sum = 0.0
                batch_labels = []
                batch_pcls = []
                batch_pexpr = []
                batch_psys = []
                batch_defer = []
                batch_weights = []

                for t in range(T):
                    f_t = self.model.cnn(imgs[:, t]).flatten(1)
                    g_cls = self.model.clf_head(f_t).view(B, K, 2)
                    if t == 0:
                        exp_prev = torch.zeros(B, K, device=self.device)
                        lstm_in = torch.cat([f_t, exp_prev], 1).unsqueeze(1)
                    else:
                        lstm_in = torch.cat([f_t, exp[:, t - 1]], 1).unsqueeze(1)
                    lstm_out, hidden = self.model.lstm(lstm_in, hidden)
                    g_def = self.model.def_head(lstm_out.squeeze(1))
                    g_def_expanded = g_def.unsqueeze(-1)
                    g_out = torch.cat([g_cls, g_def_expanded], dim=2)

                    if is_pretraining:
                        loss_sum += loss_fn(g_cls, lbs[:, t], wts[:, t])
                        sys_pred = F.softmax(g_cls, dim=2)[:, :, 1]
                        batch_pexpr.append(np.zeros((B, K), dtype=np.float32))
                        batch_defer.append(np.zeros((B, K), dtype=np.bool_))
                    else:
                        loss_sum += loss_fn(g_out, lbs[:, t], exp[:, t], wts[:, t])
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
                progress_bar.set_postfix(loss=f"Test Loss: {loss:.4f}")

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

        Y = all_labels.reshape(-1, all_labels.shape[-1])
        P = all_pcls.reshape(-1, all_pcls.shape[-1])
        E = all_pexpr.reshape(-1, all_pexpr.shape[-1])
        S = all_psys.reshape(-1, all_psys.shape[-1])
        D = all_defer.reshape(-1, all_defer.shape[-1])
        W = all_weights.reshape(-1, all_weights.shape[-1])

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
            print(f"[LSTM Eval] AUC_cls={auc_cls:.4f}, AUC_exp={auc_exp:.4f}, "
                f"AUC_sys={auc_sys:.4f}, Defer Rate={defer_rate:.4f}")
            print(f"[LSTM Eval] AUPRC_cls={auprc_cls:.4f}, AUPRC_exp={auprc_exp:.4f}, AUPRC_sys={auprc_sys:.4f}")
        else:
            defer_rate = 0.0
            print(f"[LSTM Eval] AUC_cls={auc_cls:.4f}, AUC_exp={auc_exp:.4f}, AUC_sys={auc_sys:.4f}")
            print(f"[LSTM Eval] AUPRC_cls={auprc_cls:.4f}, AUPRC_exp={auprc_exp:.4f}, AUPRC_sys={auprc_sys:.4f}")

        for i in range(K):
            dr_i = defer_rates[i] if not is_pretraining else 0.0
            print(f"Class {i:2d}: AUC_cls={auc_cls_per_class[i]:.4f}, "
                f"AUC_exp={auc_exp_per_class[i]:.4f}, AUC_sys={auc_sys_per_class[i]:.4f}, "
                f"Defer Rate={dr_i:.4f}, AUPRC_cls={auprc_cls_per_class[i]:.4f}, "
                f"AUPRC_exp={auprc_exp_per_class[i]:.4f}, AUPRC_sys={auprc_sys_per_class[i]:.4f}")
        
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
                print(f"Timestep {t:2d}:")
                print(f"Class {i:2d}: AUC_cls={auc_cls_per_class_timestep[i, t]:.4f}, "
                f"AUC_exp={auc_exp_per_class_timestep[i, t]:.4f}, AUC_sys={auc_sys_per_class_timestep[i, t]:.4f}, "
                f"Defer Rate={defer_rates_per_class_timestep[i, t]:.4f}, AUPRC_cls={auprc_cls_per_class_timestep[i, t]:.4f}, "
                f"AUPRC_exp={auprc_exp_per_class_timestep[i, t]:.4f}, AUPRC_sys={auprc_sys_per_class_timestep[i, t]:.4f}")
        
        os.makedirs(f"{self.result_dir}/{seed}", exist_ok=True)
        df.to_csv(f"{self.result_dir}/{seed}/lstm_eval_detailed-epoch-{epoch_num}-hidden-{self.model.hidden_dim}-unit-{self.model.lstm_layers}-layers-{self.pretrained_epochs}-pretrained-{self.finetuned_epochs}-finetuned-no-alpha-{self.seq_len}-steps.csv", index=False)
        
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
        os.makedirs(f"{self.result_dir}/{seed}", exist_ok=True)
        summary_path = f"{self.result_dir}/{seed}/lstm_eval_timestep_summary-epoch-{epoch_num}-hidden-{self.model.hidden_dim}-unit-{self.model.lstm_layers}-layers-{self.pretrained_epochs}-pretrained-{self.finetuned_epochs}-finetuned-no-alpha-{self.seq_len}-steps.csv"
        timestep_summary.to_csv(summary_path, index=False)
        return (running_loss / n_batch, auc_cls_per_class, auc_exp_per_class, auc_sys_per_class, defer_rates, 
                auprc_cls_per_class, auprc_exp_per_class, auprc_sys_per_class,
                auc_cls_per_class_timestep, auc_exp_per_class_timestep, auc_sys_per_class_timestep,
                auprc_cls_per_class_timestep, auprc_exp_per_class_timestep, auprc_sys_per_class_timestep,
                defer_rates_per_class_timestep)
