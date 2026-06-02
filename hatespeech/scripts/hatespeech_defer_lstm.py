import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import pandas as pd
from torchtext import data
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
from copy import deepcopy
import os
import random
import time

_HS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TWITTERAAE_MODEL = os.path.join(_HS_ROOT, "third_party", "twitteraae", "model")
vocabfile = os.environ.get("TWITTERAAE_VOCAB", os.path.join(_TWITTERAAE_MODEL, "model_vocab.txt"))
modelfile = os.environ.get("TWITTERAAE_COUNTS", os.path.join(_TWITTERAAE_MODEL, "model_count_table.txt"))
labeled_data_path = os.environ.get("HATESPEECH_LABELED_CSV", os.path.join(_HS_ROOT, "data", "labeled_data.csv"))
_DEFAULT_EXPERT_CSV = os.path.join(_HS_ROOT, "expert", "expert_curve.csv")
expert_data_path = os.environ.get("HATESPEECH_EXPERT_CURVE", _DEFAULT_EXPERT_CSV)
# If HATESPEECH_EXPERT_CURVE is unset: legacy linear schedule (1 to 0.5). If set: load that CSV.
_use_expert_csv = os.environ.get("HATESPEECH_EXPERT_CURVE") is not None

K=0; wordprobs=None; w2num=None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_filters, filter_sizes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, n_filters, kernel_size=(fs, emb_dim))
                                    for fs in filter_sizes])
        self.dropout = nn.Dropout(0.5)

    def forward(self, text_ids):
        emb = self.embedding(text_ids)
        x = emb.unsqueeze(1)
        conved  = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled  = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conved]
        feat = torch.cat(pooled, dim=1)
        return self.dropout(feat)

class CNN_LSTM_Rej(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_filters, filter_sizes, pad_idx,
                 lstm_hidden=256, lstm_layers=1, bidir=False, dropout=0.5, n_classes=3):
        super().__init__()
        self.encoder = CNNEncoder(vocab_size, emb_dim, n_filters, filter_sizes, pad_idx)
        self.feat_dim = n_filters * len(filter_sizes)

        self.lstm = nn.LSTM(input_size=self.feat_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidir,
                            dropout=0.0 if lstm_layers==1 else dropout)

        out_dim = lstm_hidden * (2 if bidir else 1)
        self.fc_cls = nn.Linear(self.feat_dim, n_classes)
        self.fc_rej = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_ids):
        if text_ids.dim() != 3:
            raise ValueError(f"Expect input [B, T, L], got {text_ids.shape}")
        B, T, L = text_ids.size()

        x2d  = text_ids.view(B*T, L).contiguous()
        feat = self.encoder(x2d) 
        feat_seq = feat.view(B, T, -1).contiguous()

        out_seq, _ = self.lstm(feat_seq)
        out_seq = self.dropout(out_seq)
        z_cls = self.fc_cls(feat_seq)
        z_rej = self.fc_rej(out_seq)

        logits = torch.cat([z_cls, z_rej], dim=2)
        logits = logits.contiguous().view(B*T, -1)
        probs = F.softmax(logits, dim=1)
        return probs

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]]).to(device)
def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    '''
    The L_{CE} loss implementation for hatespeech, identical to CIFAR implementation
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]
    rc = [n_classes] * batch_size
    rc = torch.tensor(rc)
    outputs =  -m*torch.log2( outputs[range(batch_size), rc]) - m2*torch.log2(outputs[range(batch_size), labels])   # pick the values corresponding to the labels
    return torch.sum(outputs)/batch_size

def train_reject(model, iterator, optimizer,alpha):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        batch_size = predictions.size()[0]
        # get expert predictions and costs 
        m = (batch.expert)*1.0
        m2 = [1] * batch_size
        m2 = torch.tensor(m2)
        for j in range (0,batch_size):
            exp = m[j].item()
            if exp:
                m2[j] = alpha
            else:
                m2[j] = 1

        m2 = m2.to(device)

        loss = reject_CrossEntropyLoss(predictions, m, batch.label, m2, 3)

        acc = categorical_accuracy(predictions, batch.label.to(device))
        
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate_reject(model, iterator):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            batch_size = predictions.size()[0]
            m = batch.expert
            m2 = [1] * batch_size
            m2 = torch.tensor(m2)
            m2 = m2.to(device)
            loss = reject_CrossEntropyLoss(predictions, m, batch.label, m2, 3)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def metrics_print(net, loader):
    net.eval()
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    with torch.no_grad():
        for data in loader:
            outputs = net(data.text)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]
            for i in range(0,batch_size):
                r = (predicted[i].item() == 3)
                if r==0:
                    total += 1
                    correct += (predicted[i] == data.label[i]).item()
                    correct_sys += (predicted[i] == data.label[i]).item()
                if r==1:
                    exp +=  data.expert[i].item()
                    correct_sys += data.expert[i].item()
                    exp_total+=1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
    print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def metrics_print_fairness(net, loader):
    net.eval()
    group_1 = 0
    group_1_counts = 0
    group_0 = 0
    group_0_counts = 0

    with torch.no_grad():
        for data in loader:
            outputs = net(data.text)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]
            for i in range(0,batch_size):
                r = (predicted[i].item() == 3)
                prediction = 0
                if r==0:
                    prediction = predicted[i]
                if r==1:
                    prediction = data.expertlabel[i].item()

                if  data.group[i].item() == 0:
                    if data.label[i].item() == 2:
                        group_0_counts += 1
                        if prediction == 1 or prediction ==0:
                            group_0 += 1
                else:
                    if data.label[i].item() == 2:
                        group_1_counts += 1
                        if prediction == 1 or prediction ==0:
                            group_1 += 1


    to_print={"group0":group_0/(group_0_counts+0.0001), "group1": group_1/(group_1_counts+0.0001), "discrimination":group_0/(group_0_counts+0.0001)- group_1/(group_1_counts+0.0001) }
    return [group_0/(group_0_counts+0.0001), group_1/(group_1_counts+0.0001), abs(group_0/(group_0_counts+0.0001)- group_1/(group_1_counts+0.0001))]


def load_model():
    """Idempotent"""
    global vocab,w2num,N_wk,N_k,wordprobs,N_w,K, modelfile,vocabfile
    if wordprobs is not None:
        # assume already loaded
        return

    N_wk = np.loadtxt(modelfile)
    N_w = N_wk.sum(1)
    N_k = N_wk.sum(0)
    K = len(N_k)
    wordprobs = (N_wk + 1) / N_k

    vocab = [L.split("\t")[-1].strip() for L in open(vocabfile,encoding="utf8")]
    w2num = {w:i for i,w in enumerate(vocab)}
    assert len(vocab) == N_wk.shape[0]

def infer_cvb0(invocab_tokens, alpha, numpasses):
    global K,wordprobs,w2num
    doclen = len(invocab_tokens)

    # initialize with likelihoods
    Qs = np.zeros((doclen, K))
    for i in range(0,doclen):
        w = invocab_tokens[i]
        Qs[i,:] = wordprobs[w2num[w],:]
        Qs[i,:] /= Qs[i,:].sum()
    lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

    Q_k = Qs.sum(0)
    for itr in range(1,numpasses):
        # print "cvb0 iter", itr
        for i in range(0,doclen):
            Q_k -= Qs[i,:]
            Qs[i,:] = lik[i,:] * (Q_k + alpha)
            Qs[i,:] /= Qs[i,:].sum()
            Q_k += Qs[i,:]

    Q_k /= Q_k.sum()
    return Q_k

def predict_lang(tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
    invocab_tokens = [w.lower() for w in tokens if w.lower() in w2num]
    # check that at least xx tokens are in vocabulary
    if len(invocab_tokens) < thresh1:
        return None  
    # check that at least yy% of tokens are in vocabulary
    elif len(invocab_tokens) / len(tokens) < thresh2:
        return None
    else:
        posterior = infer_cvb0(invocab_tokens, alpha=alpha, numpasses=numpasses)
        return posterior



@torch.no_grad()
def eval_metrics_dict(net, loader):
    net.eval()
    correct_clf = 0
    correct_sys = 0
    exp_correct = 0
    exp_total   = 0
    taken_total = 0
    real_total  = 0

    for data in loader:
        out = net(data.text)
        pred = out.argmax(1)
        B = pred.size(0)
        for i in range(B):
            is_rej = (pred[i].item() == 3)
            if not is_rej:
                taken_total += 1
                hit = (pred[i] == data.label[i]).item()
                correct_clf += hit
                correct_sys += hit
            else:
                e_hit = (data.expert[i].item())
                exp_correct += e_hit
                correct_sys += e_hit
                exp_total   += 1
            real_total  += 1

    coverage       = 100.0 * taken_total / max(1, real_total)
    system_acc     = 100.0 * correct_sys / max(1, real_total)
    expert_acc     = 100.0 * exp_correct / max(1, exp_total) if exp_total>0 else 0.0
    classifier_acc = 100.0 * correct_clf / max(1, taken_total) if taken_total>0 else 0.0
    return dict(
        coverage=coverage,
        system_acc=system_acc,
        expert_acc=expert_acc,
        classifier_acc=classifier_acc,
        sample_count=real_total
    )


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax()

    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
        
        out = self.fc(cat)
        out = self.softmax(out)
        return out

class CNN_rej(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.embedding_rej = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs_rej = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc_rej = nn.Linear(len(filter_sizes) * n_filters, 1)
        
        self.dropout_rej = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax()

    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
 
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        cat_rej = self.dropout_rej(torch.cat(pooled, dim = 1))

        out_rej = self.fc_rej(cat_rej)
        #cat = [batch size, n_filters * len(filter_sizes)]
        
        out = self.fc(cat)
        out =  torch.cat((out, out_rej), 1)

        out = self.softmax(out)
        return out


@torch.no_grad()
def eval_fullseq_per_t(model, TEXT, LABEL, EXPERT, buckets, seq_ids, T, batch_size):
    model.eval()

    taken_total    = np.zeros(T, dtype=np.int64)
    correct_clf    = np.zeros(T, dtype=np.int64)
    expert_total   = np.zeros(T, dtype=np.int64)
    expert_correct = np.zeros(T, dtype=np.int64)
    real_total     = np.zeros(T, dtype=np.int64)

    ids = list(seq_ids)
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i+batch_size]
        seq_batch = [buckets[sid] for sid in chunk]   # List[List[Example]]
        x_btl, y_bt, m_bt = collate_seq_to_BTL(TEXT, LABEL, EXPERT, seq_batch)
        B = x_btl.size(0)

        probs = model(x_btl)
        preds = probs.argmax(1).view(B, T)
        y     = y_bt.view(B, T)
        mexp  = m_bt.view(B, T)

        for t in range(T):
            real_total[t] += B

            pt = preds[:, t]
            yt = y[:, t]
            et = mexp[:, t]

            is_reject = (pt == 3)
            taken_total[t] += int((~is_reject).sum().item())
            if taken_total[t] > 0:
                correct_clf[t] += int((pt[~is_reject] == yt[~is_reject]).sum().item())

            expert_total[t]   += int(is_reject.sum().item())
            if expert_total[t] > 0:
                expert_correct[t] += int(et[is_reject].sum().item())

    rows = []
    for t in range(T):
        cov   = 100.0 * (taken_total[t] / max(1, real_total[t]))
        clf_a = 100.0 * (correct_clf[t] / max(1, taken_total[t])) if taken_total[t] > 0 else 0.0
        exp_a = 100.0 * (expert_correct[t] / max(1, expert_total[t])) if expert_total[t] > 0 else 0.0
        sys_a = 100.0 * ((correct_clf[t] + expert_correct[t]) / max(1, real_total[t]))

        rows.append({
            "timestep": t,
            "coverage": cov,
            "system_acc": sys_a,
            "expert_acc": exp_a,
            "classifier_acc": clf_a,
            "sample_count": int(real_total[t]),
        })
    return rows


load_model()

TEXT = data.Field(tokenize = 'spacy', batch_first = True)
LABEL = data.LabelField(dtype = torch.long, sequential=False, use_vocab=False)
EXPERT = data.LabelField(dtype = torch.long,  sequential=False, use_vocab=False)
GROUP = data.LabelField(dtype = torch.long, sequential=False, use_vocab=False)
EXPERTLABEL = data.LabelField(dtype = torch.long, sequential=False, use_vocab=False)

fields = [(None, None),(None, None),('expertlabel', EXPERTLABEL),('group', GROUP),('expert', EXPERT),
          ('label', LABEL), ('text', TEXT)]

train_data_orig = data.TabularDataset.splits(
                                        path = '',
                                        train = labeled_data_path,
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True)


seq_len = 10
p_start = 1
p_end = 0.5
q_start = 1
q_end = 0.5

exp_data = None
if _use_expert_csv:
    exp_data = pd.read_csv(expert_data_path)
    exp_data.index = exp_data.index.astype(int)
    if len(exp_data) < seq_len:
        raise ValueError(
            f"Expert curve {expert_data_path} has {len(exp_data)} rows; need at least seq_len={seq_len}"
        )

expert_tag = os.environ.get("HATESPEECH_EXPERT_TAG", "default")
time_stamp = time.strftime("%Y%m%d_%H%M%S")
base_model_root = f"model/lstm_models_new/{time_stamp}/alpha1.0_seq_len{seq_len}_{expert_tag}"

_seeds_env = os.environ.get("HATESPEECH_SEEDS", "").strip()
if _seeds_env:
    seed_list = [int(x.strip()) for x in _seeds_env.split(",") if x.strip()]
else:
    seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

if _use_expert_csv:
    print(f"[hatespeech_defer_lstm] expert CSV: {expert_data_path}")
else:
    print(f"[hatespeech_defer_lstm] expert: linear p {p_start} to {p_end} (set HATESPEECH_EXPERT_CURVE for CSV)")
print(f"[hatespeech_defer_lstm] expert_tag={expert_tag} seeds={seed_list}")

for seed in seed_list:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_root = os.path.join(base_model_root, f"seed{seed}")
    os.makedirs(model_root, exist_ok=True)

    rows = []

    all_data = train_data_orig[0]

    for (ex_idx, example) in enumerate(all_data):
        index = ex_idx % seq_len
        seq_id = ex_idx // seq_len
        if _use_expert_csv:
            p_t = exp_data.loc[index, "p"]
            q_t = exp_data.loc[index, "p"]
        else:
            frac = index / (seq_len - 1) if seq_len > 1 else 0.0
            p_t = p_start + (p_end - p_start) * frac
            q_t = q_start + (q_end - q_start) * frac
        lang = predict_lang(vars(example)['text'])
        aa = 0
        try:
                if lang[0] >= 0.5:
                    aa = 1
        except:
                print("error processing tweet: "+str(vars(example)['text']))
        label = vars(example)['label']
        exp = 0
        exp_label = 0
        aa_frac = 0
        if aa == 1: # if tweet is african american
                coin = np.random.binomial(1,p_t)
                if coin:
                    exp =1 
                    exp_label = np.long(label)
                else:
                    exp_label = np.long(np.argmax(np.random.multinomial(1,[1/3]*3 , size=1)))
                    exp =0
        else:
                coin = np.random.binomial(1,q_t)
                if coin:
                    exp =1 # is right 90% of time
                    exp_label = np.long(label)
                else:
                    exp_label = np.long(np.argmax(np.random.multinomial(1,[1/3]*3 , size=1)))
                    exp =0

        # coin = np.random.binomial(1,q_t)
        # if coin:
        #     exp =1 # is right 90% of time
        #     exp_label = np.long(label)
        # else:
        #     exp_label = np.long(np.argmax(np.random.multinomial(1,[1/3]*3 , size=1)))
        #     exp =0
        vars(all_data[ex_idx])['expertlabel'] = exp_label
        vars(all_data[ex_idx])['group'] = str(aa)
        vars(all_data[ex_idx])['expert'] = exp
        vars(all_data[ex_idx])['seq_id'] = seq_id
        vars(all_data[ex_idx])['t'] = index
        aa_frac += aa
    
    from collections import defaultdict
    buckets = defaultdict(list)
    for ex in all_data:
        buckets[vars(ex)['seq_id']].append(ex)

    full_seq_ids = [sid for sid, xs in buckets.items() if len(xs) == seq_len]
    rng = np.random.RandomState(seed)
    rng.shuffle(full_seq_ids)
    n = len(full_seq_ids)
    n_train = int(0.6 * n)
    n_valid = int(0.1 * n)
    train_ids = set(full_seq_ids[:n_train])
    valid_ids = set(full_seq_ids[n_train:n_train+n_valid])
    test_ids  = set(full_seq_ids[n_train+n_valid:])

    train_examples = [ex for sid in train_ids for ex in buckets[sid]]
    valid_examples = [ex for sid in valid_ids for ex in buckets[sid]]
    test_examples  = [ex for sid in test_ids  for ex in buckets[sid]]

    fields_for_new = all_data.fields 

    train_data = data.Dataset(train_examples, fields=fields_for_new)
    valid_data = data.Dataset(valid_examples, fields=fields_for_new)
    test_data  = data.Dataset(test_examples,  fields=fields_for_new)

    from types import SimpleNamespace

    def build_seq_buckets(dataset, seq_len):
        """Return {seq_id: [Example,...(len=seq_len, sorted by t)]} only full-length sequences"""
        buckets = defaultdict(list)
        for ex in dataset.examples:
            buckets[int(vars(ex)['seq_id'])].append(ex)
        full = {}
        for sid, xs in buckets.items():
            if len(xs) == seq_len:
                xs = sorted(xs, key=lambda e: int(vars(e)['t']))
                full[sid] = xs
        return full

    def collate_seq_to_BTL(TEXT, LABEL, EXPERT, seq_batch):
        """
        seq_batch: List[List[Example]]  B sequences, each of length T
        Returns:
        text_btl: [B, T, L]  LongTensor
        labels  : [B*T]      LongTensor
        expert  : [B*T]      FloatTensor(0/1)
        """
        texts, labels, experts = [], [], []
        for seq in seq_batch:
            for ex in seq:
                v = vars(ex)
                texts.append(v['text'])
                labels.append(v['label'])
                experts.append(v['expert'])
        text_btL = TEXT.process(texts).to(device)
        B = len(seq_batch); T = len(seq_batch[0]) if B>0 else 0
        text_btl = text_btL.view(B, T, -1).contiguous()

        labels_t = LABEL.process(labels).to(device).view(-1)
        expert_t = EXPERT.process(experts).to(device).view(-1).float()
        return text_btl, labels_t, expert_t

    def seq_loader(TEXT, LABEL, EXPERT, buckets, seq_ids, T, batch_size, shuffle):
        ids = list(seq_ids)
        if shuffle:
            random.shuffle(ids)
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i+batch_size]
            seq_batch = [buckets[sid] for sid in chunk]          # each of length T
            x_btl, y_bt, m_bt = collate_seq_to_BTL(TEXT, LABEL, EXPERT, seq_batch)
            yield SimpleNamespace(text=x_btl, label=y_bt, expert=m_bt)

    MAX_VOCAB_SIZE = 25_000
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 300
    FILTER_SIZES = [3,4,5]
    DROPOUT = 0.5
    T = seq_len

    model = CNN_LSTM_Rej(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, PAD_IDX,
                        lstm_hidden=256, lstm_layers=1, bidir=False, dropout=DROPOUT, n_classes=3).to(device)
                    
    pretrained_embeddings = TEXT.vocab.vectors
    model.encoder.embedding.weight.data.copy_(pretrained_embeddings)
    model.encoder.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.encoder.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())
    alpha = 1.0
    N_EPOCHS = 20
    BATCH_SIZE = 64

    train_buckets = build_seq_buckets(train_data, T)
    valid_buckets = build_seq_buckets(valid_data, T)
    test_buckets  = build_seq_buckets(test_data,  T)

    train_ids = list(train_buckets.keys())
    valid_ids = list(valid_buckets.keys())
    test_ids  = list(test_buckets.keys())

    print(f"#sequences -> train {len(train_ids)}, valid {len(valid_ids)}, test {len(test_ids)}")

    def train_one_epoch(model, buckets, ids):
        model.train()
        tot_loss, tot_acc, steps = 0.0, 0.0, 0
        for batch in seq_loader(TEXT, LABEL, EXPERT, buckets, ids, T, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            probs = model(batch.text)                     # input [B, T, L], output [B*T, 4]
            BT = probs.size(0)
            m  = batch.expert
            m2 = torch.where(m > 0.5,
                            torch.full((BT,), alpha, device=device),
                            torch.ones(BT, device=device))
            loss = reject_CrossEntropyLoss(probs, m, batch.label, m2, 3)
            acc  = categorical_accuracy(probs, batch.label)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_acc  += acc.item()
            steps += 1
        return (tot_loss/max(1,steps), tot_acc/max(1,steps))

    @torch.no_grad()
    def eval_seq(model, buckets, ids):
        model.eval()
        def loader():
            for b in seq_loader(TEXT, LABEL, EXPERT, buckets, ids, T, BATCH_SIZE, shuffle=False):
                yield b
        return eval_metrics_dict(model, loader())

    best_val = -1.0
    best_state = None

    for epoch in range(1, N_EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_buckets, train_ids)
        val_m = eval_seq(model, valid_buckets, valid_ids)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"valid: sys_acc={val_m['system_acc']:.2f} cov={val_m['coverage']:.2f} "
            f"clf_acc={val_m['classifier_acc']:.2f} exp_acc={val_m['expert_acc']:.2f} | "
            f"{dt:.1f}s")
        if val_m["system_acc"] >= best_val:
            best_val = val_m["system_acc"]
            best_state = copy.deepcopy(model.state_dict())

    os.makedirs(model_root, exist_ok=True)
    best_path = os.path.join(model_root, "best_model.pth")
    torch.save(best_state, best_path)
    print(f"Saved best model -> {best_path}")

    model.load_state_dict(best_state)
    test_m = eval_seq(model, test_buckets, test_ids)
    print("TEST summary:", test_m)

    per_t_rows = eval_fullseq_per_t(model, TEXT, LABEL, EXPERT, test_buckets, test_ids, T, BATCH_SIZE)
    df = (pd.DataFrame(per_t_rows)
            .sort_values("timestep")[["timestep","coverage","system_acc","expert_acc","classifier_acc","sample_count"]])
    csv_path = os.path.join(model_root, "metrics_test_fullseq_per_t.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Full-seq] Per-t metrics saved -> {csv_path}")