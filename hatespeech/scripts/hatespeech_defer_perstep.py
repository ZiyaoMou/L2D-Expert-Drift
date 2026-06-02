from __future__ import division
import numpy as np
import sys,os
import numpy as np
import torch
from torchtext import data
from torchtext import datasets
import time
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import copy
_HS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TWITTERAAE_MODEL = os.path.join(_HS_ROOT, "third_party", "twitteraae", "model")
vocabfile = os.environ.get("TWITTERAAE_VOCAB", os.path.join(_TWITTERAAE_MODEL, "model_vocab.txt"))
modelfile = os.environ.get("TWITTERAAE_COUNTS", os.path.join(_TWITTERAAE_MODEL, "model_count_table.txt"))
_DEFAULT_EXPERT_CSV = os.path.join(_HS_ROOT, "expert", "expert_curve.csv")
expert_data_path = os.environ.get("HATESPEECH_EXPERT_CURVE", _DEFAULT_EXPERT_CSV)
labeled_data_path = os.environ.get("HATESPEECH_LABELED_CSV", os.path.join(_HS_ROOT, "data", "labeled_data.csv"))

# the following functions are copied from twitteraae for convenience
K=0; wordprobs=None; w2num=None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# p_start = 1
# p_end = 0.5
# q_start = 1.0
# q_end = 0.5

exp_data = pd.read_csv(expert_data_path)
exp_data.index = exp_data.index.astype(int)
if len(exp_data) < seq_len:
    raise ValueError(
        f"Expert curve {expert_data_path} has {len(exp_data)} rows; need at least seq_len={seq_len}"
    )

expert_tag = os.environ.get("HATESPEECH_EXPERT_TAG", "default")
time_stamp = time.strftime("%Y%m%d_%H%M%S")
base_model_root = f"model/perstep_models/{time_stamp}/alpha1.0_seq_len{seq_len}_{expert_tag}"
os.makedirs(base_model_root, exist_ok=True)

_seeds_env = os.environ.get("HATESPEECH_SEEDS", "").strip()
if _seeds_env:
    seed_list = [int(x.strip()) for x in _seeds_env.split(",") if x.strip()]
else:
    seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

print(f"[hatespeech_defer_perstep] expert curve: {expert_data_path}")
print(f"[hatespeech_defer_perstep] expert_tag={expert_tag} seeds={seed_list}")

for seed in seed_list:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_root = os.path.join(base_model_root, f"seed{seed}")
    os.makedirs(model_root, exist_ok=True)

    rows = []

    for t in range(seq_len):
        # p_t = p_start + (p_end - p_start) * t / (seq_len - 1)
        # q_t = q_start + (q_end - q_start) * t / (seq_len - 1)
        # print(f"p_t: {p_t}, q_t: {q_t}")
        p_t = exp_data.loc[t, 'p']
        q_t = exp_data.loc[t, 'p']
        # build expert data
        all_data = train_data_orig[0]
        # tracker variables for statistics
        sum = 0
        total = 0
        i = 0
        aa_frac = 0
        for example in all_data:
            lang = predict_lang(vars(example)['text'])
            aa = 0
            try:
                if lang[0] >= 0.5:
                    aa = 1
            except:
                print("error processing tweet: "+str(vars(example)['text']))
            label = vars(example)['label']
            exp = 0 # 0: expert wrong, 1: expert is right
            exp_label = 0
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
            vars(all_data[i])['expertlabel'] = exp_label
            vars(all_data[i])['group'] = str(aa)
            vars(all_data[i])['expert'] = exp
            aa_frac += aa
            i += 1
            total +=1
            sum += exp
        LABEL.build_vocab(all_data)
        EXPERT.build_vocab(all_data)
        GROUP.build_vocab(all_data)
        EXPERTLABEL.build_vocab(all_data)
        MAX_VOCAB_SIZE = 25_000

        TEXT.build_vocab(all_data, 
                        max_size = MAX_VOCAB_SIZE, 
                        vectors = "glove.6B.100d", 
                        unk_init = torch.Tensor.normal_)

        print("TEXT vocab size =", len(TEXT.vocab))
        print("PAD in stoi?     ", "<pad>" in TEXT.vocab.stoi)

        train_data, test_data, valid_data  = all_data.split(split_ratio=[0.6,0.1,0.3])

        BATCH_SIZE = 64

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            sort = False,
            batch_size = BATCH_SIZE, 
            device = device)

        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100 # fixed
        N_FILTERS = 300 # hyperparameterr
        FILTER_SIZES = [3,4,5]
        OUTPUT_DIM = 4
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = CNN_rej(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 3, DROPOUT, PAD_IDX)

        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        model = model.to(device)
        alpha = 1.0
        N_EPOCHS = 5

        best_valid_loss = 0
        best_model = None
        for epoch in range(N_EPOCHS):

            start_time = time.time()
            train_loss, train_acc = train_reject(model, train_iterator, optimizer, alpha)

            valid_loss = metrics_print(model,valid_iterator)[1]

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss >= best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(model)
        
        t_model_path = os.path.join(model_root, f"t{t}_best_model.pth")
        torch.save(best_model.state_dict(), t_model_path)
        print(f"  [seed {seed}] saved best model for t={t} -> {t_model_path}")
        metrics_print(best_model, test_iterator)
        test_metrics = eval_metrics_dict(best_model, test_iterator)
        rows.append({
            "timestep": t,
            "coverage": test_metrics["coverage"],
            "system_acc": test_metrics["system_acc"],
            "expert_acc": test_metrics["expert_acc"],
            "classifier_acc": test_metrics["classifier_acc"],
            "sample_count": test_metrics["sample_count"],
        })

    df = (pd.DataFrame(rows)
                .sort_values("timestep")[["timestep","coverage","system_acc","expert_acc","classifier_acc","sample_count"]])
    csv_path = os.path.join(model_root, "epoch5_metrics_test.csv")
    df.to_csv(csv_path, index=False)
    print(f"[seed {seed}] metrics saved -> {csv_path}")
            

