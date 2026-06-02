import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import os
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.loss import compute_AUROC

use_gpu = torch.cuda.is_available()

class DenseNet121_defer(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, out_size):
        super(DenseNet121_defer, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size * 3) # for CE use 2, for LCE use 3
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def CrossEntropyLoss_defer(outputs, labels, expert_costs, weights, alpha):
    '''
    outputs: model outputs
    labels: target vector
    expert_costs: expert predictions (agreement with label) 
    weights: uncertainty flagging
    alpha: vector of alphas
    '''
    # expert_costs: for now expert prediction raw
    batch_size = outputs.size()[0]  # batch_size
    total_loss = 0
    j = 0
    reject_class = [2] * batch_size
    for i in range(14):
        out_softmax = torch.nn.functional.softmax(outputs[range(batch_size),j:j+3])
        j += 3
        cost_i = expert_costs[range(batch_size),i] == labels[range(batch_size),i]
        opp_cost =  expert_costs[range(batch_size),i] != labels[range(batch_size),i]
        opp_cost = opp_cost.float()
        cost_i = cost_i.float()
        loss_defer = -weights[range(batch_size),i] * cost_i * torch.log2(out_softmax[range(batch_size), reject_class])
        loss_class = -(alpha[i]*cost_i + opp_cost) * weights[range(batch_size),i] * torch.log2(out_softmax[range(batch_size), labels[range(batch_size),i].long()])
        loss_total = torch.sum(loss_class + loss_defer)/(torch.sum(weights[range(batch_size),i] ) +0.000000001)
        total_loss += loss_total
    total_loss /= 14
    return total_loss

def CrossEntropyLoss_no_defer(outputs, labels, expert_costs, weights, alpha):
    '''
    outputs: [B, 42] (each 3 for [class0, class1, defer])
    labels:  [B, 14]
    expert_costs: [B, 14] (unused)
    weights: [B, 14]
    alpha: unused
    '''
    B = outputs.size(0)
    K = 14
    outputs = outputs.view(B, K, 3)
    probs = torch.nn.functional.softmax(outputs, dim=2)
    labels_long = labels.long()
    correct_probs = probs.gather(2, labels_long.unsqueeze(2)).squeeze(2)
    log_probs = torch.log2(correct_probs + 1e-12)
    weighted_loss = -weights * log_probs
    class_losses = weighted_loss.sum(dim=0) / (weights.sum(dim=0) + 1e-12)
    total_loss = class_losses.mean()
    return total_loss

def CrossEntropyLoss_with_defer(outputs, labels, expert_costs, weights, alpha):
    '''
    outputs: [B, 14, 3] (class0, class1, defer)
    labels:  [B, 14]
    expert_costs: [B, 14] (0/1)
    weights: [B, 14]
    alpha: [14] or None
    '''
    B, K, _ = outputs.shape
    probs = torch.nn.functional.softmax(outputs, dim=2)
    labels_long = labels.long()
    p_true = probs.gather(2, labels_long.unsqueeze(2)).squeeze(2)
    p_defer = probs[..., 2]
    expert_correct = (expert_costs == labels).float()
    expert_wrong = 1.0 - expert_correct
    if alpha is None:
        alpha = torch.ones(K, device=outputs.device)
    alpha = alpha.view(1, K)
    w_cls = alpha * expert_correct + expert_wrong
    if weights is None:
        weights = torch.ones_like(labels, dtype=torch.float32)
    loss_cls = - w_cls * weights * torch.log(p_true + 1e-12)
    loss_def = - weights * expert_correct * torch.log(p_defer + 1e-12)
    denom = weights.sum(dim=0).clamp(min=1e-12)
    cls_per_class = loss_cls.sum(dim=0) / denom
    def_per_class = loss_def.sum(dim=0) / denom
    return (cls_per_class + def_per_class).mean()

class CheXpertTrainer_defer():
    def train_defer(model, rad_index, learn_to_defer, dataLoadertrain, dataLoaderVal, nnClassCount, trMaxEpoch, launchTimestamp, alpha, checkpoint):
            
            #SETTINGS: OPTIMIZER & SCHEDULER
            optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
            scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, mode='min')

            #SETTINGS: LOSS
            loss = 0
            #LOAD CHECKPOINT 
            if checkpoint != None and use_gpu and os.path.exists(checkpoint):
                modelCheckpoint = torch.load(checkpoint)
                # Handle different checkpoint formats
                if isinstance(modelCheckpoint, dict):
                    if 'state_dict' in modelCheckpoint:
                        state_dict = modelCheckpoint['state_dict']
                        # Handle DataParallel weights
                        if any(key.startswith('module.') for key in state_dict.keys()):
                            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                        model.load_state_dict(state_dict)
                        if 'optimizer' in modelCheckpoint:
                            optimizer.load_state_dict(modelCheckpoint['optimizer'])
                    else:
                        # Direct state_dict format
                        state_dict = modelCheckpoint
                        # Handle DataParallel weights
                        if any(key.startswith('module.') for key in state_dict.keys()):
                            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                        model.load_state_dict(state_dict)
                else:
                    # Direct state_dict format
                    state_dict = modelCheckpoint
                    # Handle DataParallel weights
                    if any(key.startswith('module.') for key in state_dict.keys()):
                        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                    model.load_state_dict(state_dict)
                return 0, 0, 0
            
            #TRAIN THE NETWORK
            else: 
                lossMIN = 100000
                
                for epochID in range(0, trMaxEpoch):
                    
                    timestampTime = time.strftime("%H%M%S")
                    timestampDate = time.strftime("%d%m%Y")
                    timestampSTART = timestampDate + '-' + timestampTime
                    
                    batchs, losst, losse = CheXpertTrainer_defer.epochTrain_defer(model, rad_index, learn_to_defer, dataLoadertrain, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss, alpha)
                    lossVal = 0
                    #lossVal = CheXpertTrainer_defer.epochVal_defer(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss, alpha)
                    #test_defer_metrics(model, dataLoaderVal)      

                    timestampTime = time.strftime("%H%M%S")
                    timestampDate = time.strftime("%d%m%Y")
                    timestampEND = timestampDate + '-' + timestampTime
                    scheduler.step(lossVal)
                    if lossVal < lossMIN:
                        lossMIN = lossVal    
                        #torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'mod_defer-epoch'+str(epochID)+'-' + launchTimestamp + '_min' + '.pth.tar')
                        print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
                    else:
                        #torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossVal, 'optimizer' : optimizer.state_dict()}, 'mod_defer-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar')
                        print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                return batchs, losst, losse        


    def epochTrain_defer(model,rad_index, learn_to_defer, dataLoader,dataloaderVal, optimizer, epochMax, classCount, loss, alpha):
        
        batch = []
        losstrain = []
        losseval = []

        model.train()
        for batchID, (varInput, target, rad1, rad2, rad3, weights) in enumerate(tqdm(dataLoader, desc="Training", leave=False)): # change to dataloader regular

            varInput = varInput.cuda(non_blocking = True)
            varTarget = target.cuda(non_blocking = True)
            varRad = 0
            if rad_index == 1:
                varRad = rad1.cuda(non_blocking= True)
            elif rad_index == 2:
                varRad = rad2.cuda(non_blocking= True)
            elif rad_index == 3:
                varRad = rad3.cuda(non_blocking= True)
            else:
                raise Exception('Invalid rad index')
            varOutput = model(varInput)
            lossvalue = 0
            if not learn_to_defer:
                lossvalue = CrossEntropyLoss_no_defer(varOutput, varTarget,varRad, weights.cuda(non_blocking=True),alpha )
            else:
                lossvalue = CrossEntropyLoss_defer(varOutput, varTarget,varRad, weights.cuda(non_blocking=True),alpha )
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)

        return batch, losstrain, losseval

    def epochVal_defer(model, dataLoader, learn_to_defer, optimizer, epochMax, classCount, loss, alpha):
            
            model.eval()
            
            lossVal = 0
            lossValNorm = 0

            with torch.no_grad():
                for i, (varInput, target, rad1, rad2, rad3, weights) in enumerate(tqdm(dataLoader, desc="Validation", leave=False)):
                    
                    varInput = varInput.cuda(non_blocking = True)
                    target = target.cuda(non_blocking = True)
                    varOutput = model(varInput)
                    losstensor = 0
                    if not learn_to_defer:
                        losstensor = CrossEntropyLoss_no_defer(varOutput, varTarget,varRad, weights.cuda(non_blocking=True),alpha )
                    else:
                        losstensor = CrossEntropyLoss_defer(varOutput, varTarget,varRad, weights.cuda(non_blocking=True),alpha )
        
                    lossVal += losstensor
                    lossValNorm += 1
                    
            outLoss = lossVal / lossValNorm
            return outLoss
    

    def test_defer(model, dataLoaderTest, nnClassCount, checkpoint, class_names):   
        
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        with torch.no_grad():
            coverages = [0] *14
            for i, (input, target, rad1, rad2, rad3, weights) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).cuda()

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w).cuda(non_blocking = True)
            
                out = model(varInput)
                batch_size = out.size()[0]  # batch_size
                j = 0
                batch_predictions = torch.FloatTensor(batch_size, 14).cuda()
                for k in range(14):
                    out_softmax = torch.nn.functional.softmax(out[range(batch_size),j:j+3])
                    out_softmax_class = torch.nn.functional.softmax(out[range(batch_size),j:j+2])
                    _, predicted = torch.max(out_softmax.data, 1)
                    sys_output = [0] * batch_size
                    for idx in range(batch_size):
                        if predicted[idx] == 2:
                            sys_output[idx] = rad1[idx][k]
                            batch_predictions[idx, k] = rad1[idx][k]
                        else:
                            coverages[k] += 1
                            # print(out_softmax_class[idx][1].detach().cpu().numpy())
                            sys_output[idx] = out_softmax_class[idx][1].detach().cpu().numpy()
                            batch_predictions[idx, k] = out_softmax_class[idx][1]
                    j += 3

                outPRED = torch.cat((outPRED, batch_predictions), 0)

        aurocIndividual = compute_AUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        print(coverages)
        return outGT, outPRED

    def test_epoch_defer(self, model, loader, alpha, device, rad_index=1, use_defer=True):
        model.eval()
        running_loss = 0.0

        all_labels  = []
        all_pcls    = []
        all_pexpr   = []
        all_psys    = []
        all_defer   = []
        all_weights = []
        total_defer = 0
        total_preds = 0

        with torch.no_grad():
            for images, labels, rad1, rad2, rad3, weights in tqdm(loader, desc="Testing", leave=False):
                images  = images.to(device)
                labels  = labels.to(device)
                weights = weights.to(device)

                rad = {1: rad1, 2: rad2, 3: rad3}.get(rad_index)
                if rad is None:
                    raise ValueError("Invalid rad_index")
                rad = rad.to(device)

                B, K = labels.shape
                out = model(images)
                g   = out.view(B, K, 3)

                p_three = torch.softmax(g, dim=2)
                p_cls_0 = p_three[..., 0]
                p_cls_1 = p_three[..., 1]
                p_defer = p_three[..., 2]

                g_cls         = g[..., :2]
                p_cls_2class  = torch.softmax(g_cls, dim=2)  
                p_cls_1_sep   = p_cls_2class[..., 1]

                if use_defer:
                    loss = CrossEntropyLoss_defer(out, labels, rad, weights, alpha)
                else:
                    loss = CrossEntropyLoss_no_defer(out, labels, rad, weights, alpha)
                running_loss += loss.item()

                if use_defer:
                    max_cls_prob = torch.max(p_cls_0, p_cls_1)
                    mask = p_defer > max_cls_prob
                    sys_pred = torch.where(mask, rad, p_cls_1_sep)
                    all_defer.append(mask.cpu().numpy())
                    total_defer += mask.sum().item()
                    total_preds += mask.numel()
                else:
                    sys_pred = p_cls_1_sep

                all_labels.append(labels.cpu().numpy())
                all_pcls  .append(p_cls_1_sep.cpu().numpy())
                all_psys  .append(sys_pred.cpu().numpy())
                all_pexpr .append(rad.cpu().numpy() if use_defer else np.zeros_like(p_cls_1_sep.cpu().numpy()))
                all_weights.append(weights.cpu().numpy())

        Y = np.concatenate(all_labels,  axis=0)
        P = np.concatenate(all_pcls,    axis=0)
        S = np.concatenate(all_psys,    axis=0)
        E = np.concatenate(all_pexpr,   axis=0)
        W = np.concatenate(all_weights, axis=0)
        D = np.concatenate(all_defer,   axis=0) if use_defer else np.zeros_like(S)

        K = Y.shape[1]
        auc_cls_per_class = np.full(K, np.nan)
        auc_exp_per_class = np.full(K, np.nan)
        auc_sys_per_class = np.full(K, np.nan)
        auprc_cls_per_class = np.full(K, np.nan)
        auprc_exp_per_class = np.full(K, np.nan)
        auprc_sys_per_class = np.full(K, np.nan)
        for i in range(K):
            mask_i = W[:, i] == 1
            if mask_i.sum() > 1 and len(np.unique(Y[mask_i, i])) > 1:
                auc_cls_per_class[i] = roc_auc_score(Y[mask_i, i], P[mask_i, i])
                auc_exp_per_class[i] = roc_auc_score(Y[mask_i, i], E[mask_i, i])
                auc_sys_per_class[i] = roc_auc_score(Y[mask_i, i], S[mask_i, i])
                auprc_cls_per_class[i] = average_precision_score(Y[mask_i, i], P[mask_i, i])
                auprc_exp_per_class[i] = average_precision_score(Y[mask_i, i], E[mask_i, i])
                auprc_sys_per_class[i] = average_precision_score(Y[mask_i, i], S[mask_i, i])

        auc_cls = np.nanmean(auc_cls_per_class)
        auc_exp = np.nanmean(auc_exp_per_class)
        auc_sys = np.nanmean(auc_sys_per_class)
        auprc_cls = np.nanmean(auprc_cls_per_class)
        auprc_exp = np.nanmean(auprc_exp_per_class)
        auprc_sys = np.nanmean(auprc_sys_per_class)
        defer_rates = np.full(K, np.nan)
        for i in range(K):
            defer_rates[i] = D[:, i].mean() if use_defer else 0.0
        defer_rate = defer_rates.mean()

        if use_defer:
            defer_rate = total_defer / total_preds
            print(f"[Per-Step Eval] AUC_cls={auc_cls:.4f}, AUC_exp={auc_exp:.4f}, "
                f"AUC_sys={auc_sys:.4f}, Defer Rate={defer_rate:.4f}")
            print(f"[Per-Step Eval] AUPRC_cls={auprc_cls:.4f}, AUPRC_exp={auprc_exp:.4f}, AUPRC_sys={auprc_sys:.4f}")
        else:
            defer_rate = 0.0
            print(f"[Per-Step Eval] AUC_cls={auc_cls:.4f}, AUC_exp={auc_exp:.4f}, AUC_sys={auc_sys:.4f}")
            print(f"[Per-Step Eval] AUPRC_cls={auprc_cls:.4f}, AUPRC_exp={auprc_exp:.4f}, AUPRC_sys={auprc_sys:.4f}")

        for i in range(K):
            dr_i = D[:, i].mean() if use_defer else 0.0
            print(f"Class {i:2d}: AUC_cls={auc_cls_per_class[i]:.4f}, "
                f"AUC_exp={auc_exp_per_class[i]:.4f}, AUC_sys={auc_sys_per_class[i]:.4f}, "
                f"Defer Rate={dr_i:.4f}")
            print(f"Class {i:2d}: AUPRC_cls={auprc_cls_per_class[i]:.4f}, "
                f"AUPRC_exp={auprc_exp_per_class[i]:.4f}, AUPRC_sys={auprc_sys_per_class[i]:.4f}")

        return running_loss / len(loader), auc_cls_per_class, auc_exp_per_class, auc_sys_per_class, defer_rates, auprc_cls_per_class, auprc_exp_per_class, auprc_sys_per_class
