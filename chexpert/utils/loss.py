import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_AUROC(dataGT, dataPRED, classCount):
    """
    dataGT: Tensor or ndarray of shape [N, K]
    dataPRED: Tensor or ndarray of shape [N, K]
    classCount: int, number of classes
    """
    if hasattr(dataGT, 'cpu'):
        dataGT = dataGT.cpu().numpy()
    if hasattr(dataPRED, 'cpu'):
        dataPRED = dataPRED.cpu().numpy()

    outAUROC = []
    for i in range(classCount):
        try:
            auc = roc_auc_score(dataGT[:, i], dataPRED[:, i])
            outAUROC.append(auc)
        except ValueError as e:
            outAUROC.append(np.nan)  # In case only one class present
    return outAUROC

def CrossEntropyLoss_defer(outputs, labels, expert_costs, weights, alpha):
    B = outputs.size(0)
    total_loss = 0.0
    j = 0
    for i in range(14):
        logits_i = outputs[:, j : j + 3]           # [B,3]
        probs   = torch.softmax(logits_i, dim=1)   # [B,3]
        j += 3

        correct = (expert_costs[:, i] == labels[:, i]).float()
        wrong   = 1.0 - correct

        # defer-loss
        ld = - weights[:, i] * correct * torch.log(probs[:, 2] + 1e-12)
        # classify-loss
        lc = - weights[:, i] * (alpha[i] * correct + wrong) * torch.log(
                 probs[range(B), labels[:, i].long()] + 1e-12
             )

        total_loss += (ld + lc).sum() / (weights[:, i].sum() + 1e-12)
    return total_loss / 14.0

def CrossEntropyLoss_no_defer(outputs, labels, expert_costs, weights, alpha):
    '''
    outputs: model outputs
    labels: target vector
    expert_costs: expert predictions 
    weights: uncertainty flagging
    alpha: vector of alphas, not used
    '''
    # expert_costs: for now expert prediction raw
    batch_size = outputs.size()[0]  # batch_size
    total_loss = 0
    j = 0
    reject_class = [2] * batch_size
    for i in range(14):
        out_softmax = torch.nn.functional.softmax(outputs[range(batch_size),j:j+3])
        j += 3 # index variable update, for CE use 2, for LCE use 3
        loss_class = - weights[range(batch_size),i] * torch.log2(out_softmax[range(batch_size), labels[range(batch_size),i].long()])
        loss_total = torch.sum(loss_class)/(torch.sum(weights[range(batch_size),i] ) +0.000000001) # average loss
        total_loss += loss_total # total loss is the sum of all class losses
    total_loss /= 14
    return total_loss


def CrossEntropyLoss_sequence(outputs, labels, weights):
    """
    outputs: [B, T, K*2]
    labels:  [B, T, K]
    weights: [B, T, K]
    """
    B, T, D = outputs.shape
    K = D // 2
    total_loss = 0.0

    for t in range(T):
        out = outputs[:, t]      # [B, K*2]
        lbl = labels[:, t]       # [B, K]
        wts = weights[:, t]      # [B, K]

        batch_size = out.size(0)
        j = 0
        loss_t = 0
        for i in range(K):
            out_softmax = torch.nn.functional.softmax(out[:, j:j+2], dim=1)
            label_i = lbl[:, i].long()
            weight_i = wts[:, i]
            # Avoid log(0) with epsilon
            loss_class = - weight_i * torch.log2(out_softmax[range(batch_size), label_i] + 1e-7)
            loss_class = torch.sum(loss_class) / (torch.sum(weight_i) + 1e-8)
            loss_t += loss_class
            j += 2
        total_loss += loss_t / K
    return total_loss / T


def CrossEntropyLoss(outputs, labels, weights):
    # m: expert costs, labels: ground truth, n_classes: number of classes
    '''
    outputs: model outputs
    labels: target vector
    weights: uncertainty flagging
    '''
    batch_size = outputs.size()[0]  # batch_size
    total_loss = 0
    j = 0
    for i in range(14):
        out_softmax = torch.nn.functional.softmax(outputs[range(batch_size),j:j+2])
        j += 2 # index variable update, for CE use 2, for LCE use 3
        loss_class = - weights[range(batch_size),i] * torch.log2(out_softmax[range(batch_size), labels[range(batch_size),i].long()])   
        loss_class = torch.sum(loss_class)/(torch.sum(weights[range(batch_size),i] ) +0.000000001)  # average loss
        total_loss += loss_class # total loss is the sum of all class losses
    total_loss /= 14
    return total_loss
