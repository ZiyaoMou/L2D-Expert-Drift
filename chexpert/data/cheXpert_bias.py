import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.misc import get_patient_names
from torchvision import transforms
import random
import numpy as np

class CheXpertDataSet_rad_bias_seq(Dataset):
    def __init__(self,
                 image_list_file,
                 expert_model,
                 patient_names,
                 root_dir=None,
                 transform=None,
                 ignore_uncertain=True,
                 policy="ones",
                 seq_len=20,
                 step=1):
        if root_dir is None:
            root_dir = os.path.dirname(image_list_file)

        K = 14
        data_all = []

        with open(image_list_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                path = os.path.join(root_dir, line[0])
                pid = line[0].split("/")[2]
                if pid not in patient_names:
                    continue
                label, weight = line[5:], [1] * K
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1: # uncertain label
                            if ignore_uncertain:
                                weight[i] = 0
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                data_all.append((path, label, weight))

        # Shuffle and construct sequences
        random.shuffle(data_all)

        self.image_seqs = []
        self.label_seqs = []
        self.weight_seqs = []

        for start in range(0, len(data_all) - seq_len + 1, step):
            segment = data_all[start:start + seq_len]
            paths, labels, weights = zip(*segment)
            self.image_seqs.append(paths)
            self.label_seqs.append(labels)
            self.weight_seqs.append(weights)

        self.transform = transform
        self.seq_len = seq_len
        self.step = step
        self.K = K
        self.expert_model = expert_model

        # Compute expert predictions for each sequence
        self.exp_preds = []
        for label_seq in self.label_seqs:
            labels_seq = np.array(label_seq)
            preds_seq = []
            for t in range(len(label_seq)):
                y_t = labels_seq[t].reshape(1, -1)
                pred_t = self.expert_model.predict(y_t, timestep=t)
                preds_seq.append(pred_t.squeeze(0))
            self.exp_preds.append(np.stack(preds_seq, axis=0))

    def __len__(self):
        return len(self.image_seqs)

    def __getitem__(self, idx):
        imgs, lbs, wts = [], [], []
        for j in range(self.seq_len):
            img = Image.open(self.image_seqs[idx][j]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            lbs.append(self.label_seqs[idx][j])
            wts.append(self.weight_seqs[idx][j])
        imgs = torch.stack(imgs, dim=0)  # [T, C, H, W]
        lbs = torch.FloatTensor(lbs)     # [T, K]
        wts = torch.FloatTensor(wts)     # [T, K]
        rad_1 = self.exp_preds[idx]
        return imgs, lbs, wts, rad_1, rad_1, rad_1

def split_dataset_seq(train_size=0.8, random_seed=66,
                      root_dir=None, pathFileTrain=None, pathFileValid=None,
                      exp_fake=None, trBatchSize=16, seq_len=10, step=1):

    dataset_all_names = get_patient_names(pathFileTrain)
    patients_train, patients_test = train_test_split(list(dataset_all_names.keys()), test_size=0.1, random_state=random_seed)
    patients_train, patients_val = train_test_split(patients_train, test_size=0.1, random_state=random_seed)
    patients_train_leftout, patients_train = train_test_split(patients_train, test_size=train_size, random_state=random_seed)

    patients_train_dict = {p: 1 for p in patients_train}
    patients_val_dict = {p: 1 for p in patients_val}
    patients_test_dict = {p: 1 for p in patients_test}

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = CheXpertDataSet_rad_bias_seq(pathFileTrain, expert_model=exp_fake,
                                                 patient_names=patients_train_dict,
                                                 root_dir=root_dir, transform=transform_train,
                                                 seq_len=seq_len, step=step)
    dataset_val = CheXpertDataSet_rad_bias_seq(pathFileTrain, expert_model=exp_fake,
                                                 patient_names=patients_val_dict,
                                                 root_dir=root_dir, transform=transform_test,
                                                 seq_len=seq_len, step=step)
    dataset_test = CheXpertDataSet_rad_bias_seq(pathFileTrain, expert_model=exp_fake,
                                                 patient_names=patients_test_dict,
                                                 root_dir=root_dir, transform=transform_test,
                                                 seq_len=seq_len, step=step)

    dataLoaderTrain = DataLoader(dataset_train, batch_size=trBatchSize, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    dataLoaderVal = DataLoader(dataset_val, batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    dataLoaderTest = DataLoader(dataset_test, batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    print("Number of sequence samples:")
    print(f' train: {len(dataset_train)}, val: {len(dataset_val)}, test: {len(dataset_test)}')

    return dataLoaderTrain, dataLoaderVal, dataLoaderTest