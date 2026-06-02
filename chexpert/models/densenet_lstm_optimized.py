import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
from utils.loss import compute_AUROC
import pandas as pd
import os
from torch.utils.checkpoint import checkpoint

# alpha: per-pathology weights; default is ones on the same device as outputs
def l2d_loss(outputs, labels, expert_preds, weights, alpha=None):
    """
    Soft Learn-to-Defer loss, vectorized.
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
    B, K = labels.shape
    p_cls = F.softmax(g_cls, dim=2)
    total_loss = 0.0
    for i in range(K):
        batch_indices = torch.arange(B, device=labels.device)
        p_class_i = p_cls[:, i, :][batch_indices, labels[:, i].long()]
        loss_clf = -weights[:, i] * torch.log(p_class_i + 1e-12)
        loss_total_i = loss_clf.sum() / (weights[:, i].sum() + 1e-8)
        total_loss += loss_total_i
    return total_loss

class DenseLSTMDeferOptimized(nn.Module):
    def __init__(self, num_classes, lstm_hidden=512, lstm_layers=1):
        super().__init__()
        # Use weights instead of pretrained to avoid deprecation warning
        backbone = torchvision.models.densenet121(weights='IMAGENET1K_V1')
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

    def extract_features_batch(self, x_batch):
        """Extract features from a batch of images with gradient checkpointing"""
        return checkpoint(self.cnn, x_batch, use_reentrant=False)

    def forward(self, x_seq, h_prev_s=None, pretraining=False):
        B, T, C, H, W = x_seq.shape
        x = x_seq.view(B*T, C, H, W)
        
        # Use gradient checkpointing for CNN feature extraction
        f = self.extract_features_batch(x).flatten(1)
        f = f.view(B, T, -1)
        
        g_cls = self.clf_head(f)
        lstm_in = torch.cat([f, h_prev_s], dim=2)
        lstm_out, _ = self.lstm(lstm_in)
        g_def = self.def_head(lstm_out)
        g_all = torch.cat([g_cls.view(B, self.num_classes, 2), g_def.unsqueeze(-1)], dim=2).view(B, -1)
        return g_all

class CheXpertTrainerDeferLSTMOptimized:
    def __init__(self, model, expert, device, lr=0.0001, pretrained_epochs=3, finetuned_epochs=10):
        self.model = model
        self.expert = expert
        self.device = device
        self.lr = lr
        self.pretrained_epochs = pretrained_epochs
        self.finetuned_epochs = finetuned_epochs

    def train_defer_lstm(self, train_loader, test_loader):
        self.model.train()
        pretrained_model = f"checkpoints/lstm/densenet_lstm_fatigue_pretrained_1-{self.model.hidden_dim}-unit-{self.model.lstm_layers}-layers-100-steps.pth"
        if os.path.exists(pretrained_model):
            self.model.load_state_dict(torch.load(pretrained_model))
            print(f"Loaded pretrained model from {pretrained_model}")
        else:
            print(f"Pretrained model not found at {pretrained_model}")
            # Pretraining phase
            for epoch in range(self.pretrained_epochs):
                print(f"Pretraining Epoch {epoch+1}/{self.pretrained_epochs}")
                optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
                
                train_loss = self.train_epoch_optimized(train_loader, optimizer, l2d_loss_no_defer)
                print(f"Pretrain Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
                
                # Clear cache after each epoch
                torch.cuda.empty_cache()
                scheduler.step(train_loss)

        # Fine-tuning phase
        for epoch in range(self.finetuned_epochs):
            print(f"Finetuning Epoch {epoch+1}/{self.finetuned_epochs}")
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr/10)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            
            train_loss = self.train_epoch_optimized(train_loader, optimizer, l2d_loss)
            print(f"Finetune Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
            
            # Clear cache after each epoch
            torch.cuda.empty_cache()
            scheduler.step(train_loss)

    def train_epoch_optimized(self, train_loader, optimizer, loss_fn):
        self.model.train()
        running_loss = 0.0
        n_batch = 0
        progress_bar = tqdm(train_loader, leave=False, ncols=100)
        
        # Use gradient accumulation to reduce memory usage
        accumulation_steps = 2  # Accumulate gradients over 2 steps
        
        for batch_idx, (imgs, lbs, wts, rad_1, rad_2, rad_3) in enumerate(progress_bar):
            imgs = imgs.to(self.device)
            lbs = lbs.to(self.device)
            wts = wts.to(self.device)
            rad_1 = rad_1.to(self.device)
            rad_2 = rad_2.to(self.device)
            rad_3 = rad_3.to(self.device)
            B, T, K = lbs.shape
            is_pretraining = loss_fn == l2d_loss_no_defer
            
            # Process in smaller chunks to reduce memory usage
            chunk_size = min(T, 20)  # Process at most 20 timesteps at once
            total_loss = 0.0
            num_chunks = 0
            
            for chunk_start in range(0, T, chunk_size):
                chunk_end = min(chunk_start + chunk_size, T)
                chunk_T = chunk_end - chunk_start
                
                # Clear cache before processing each chunk
                torch.cuda.empty_cache()
                
                # Extract features for this chunk
                img_chunk = imgs[:, chunk_start:chunk_end]
                
                hidden = None
                chunk_loss = 0.0
                
                # Process timesteps within chunk
                for t in range(chunk_T):
                    with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
                        # Extract features with gradient checkpointing
                        f_t = checkpoint(self.model.cnn, img_chunk[:, t], use_reentrant=False).flatten(1)
                        g_cls = self.model.clf_head(f_t).view(B, K, 2)
                        
                        if chunk_start + t == 0:
                            exp_prev = torch.zeros(B, K, device=self.device)
                            lstm_in = torch.cat([f_t, exp_prev], 1).unsqueeze(1)
                        else:
                            global_t = chunk_start + t
                            lstm_in = torch.cat([f_t, rad_1[:, global_t-1]], 1).unsqueeze(1)
                        
                        lstm_out, hidden = self.model.lstm(lstm_in, hidden)
                        g_def = self.model.def_head(lstm_out.squeeze(1))
                        g_def_expanded = g_def.unsqueeze(-1)
                        g_out = torch.cat([g_cls, g_def_expanded], dim=2)
                        
                        if is_pretraining:
                            loss_t = loss_fn(g_cls, lbs[:, chunk_start + t], wts[:, chunk_start + t])
                        else:
                            loss_t = loss_fn(g_out, lbs[:, chunk_start + t], rad_1[:, chunk_start + t], wts[:, chunk_start + t])
                        
                        chunk_loss += loss_t
                
                total_loss += chunk_loss / chunk_T
                num_chunks += 1
                
                # Detach hidden state to prevent gradient accumulation across chunks
                if hidden is not None:
                    hidden = (hidden[0].detach(), hidden[1].detach())
            
            loss = total_loss / num_chunks
            
            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            running_loss += loss.item() * accumulation_steps
            n_batch += 1
            avg_loss = running_loss / n_batch
            progress_bar.set_postfix(loss=f"Train Loss: {avg_loss:.4f}")
        
        # Final optimizer step if needed
        if n_batch % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        return running_loss / n_batch 