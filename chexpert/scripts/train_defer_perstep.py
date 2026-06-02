import sys
import os
import torch
import time
import pandas as pd
from tqdm import tqdm
import argparse
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from data.cheXpert_random import split_dataset_shuffle
from data.cheXpert import split_dataset
from experts.fake import ExpertModel_fake
from models.densenet_defer import CheXpertTrainer_defer, DenseNet121_defer

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=66, help="random seed")
args = parser.parse_args()
set_all_seeds(args.seed)

nnClassCount = 14
seq_len      = 10
step         = 10
batch_size   = 16
train_ratio  = 0.01

random_seed  = args.seed

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_data = os.path.join(script_dir, "datasets")
rootDir = os.environ.get("CHEXPERT_DATA_ROOT", _default_data)
pathFileTrain = os.path.join(rootDir, "CheXpert-v1.0-small", "train.csv")
pathFileValid = os.path.join(rootDir, "CheXpert-v1.0-small", "valid.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

alpha = [1]*14
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

pretrained_model = os.environ.get(
    "CHEXPERT_PRETRAINED_STEP",
    os.path.join(script_dir, "checkpoints", "pretrained_step_model_full_pretrained_no_alpha.pth"),
)
# Expert accuracy from 1.0 to 0.7 over 10 steps (linear decay)
expert_acc = [1.0 - (0.3 * i / 9) for i in range(10)]  # [1.0, 0.967, 0.933, 0.9, 0.867, 0.833, 0.8, 0.767, 0.733, 0.7]

# Main training loop with progress bar
df = pd.DataFrame()
set_all_seeds(random_seed)
for t in tqdm(range(0, 10), desc="Training steps", unit="step"):
    
    expert_t = ExpertModel_fake(
        confounding_class=13,
        p_confound=expert_acc[t],
        p_nonconfound=expert_acc[t]
    )
    
    trainer_t = CheXpertTrainer_defer()
    dataLoaderTrain_t, dataLoaderVal_t, dataLoaderTest_t, _, _ = split_dataset(
        exp_fake=expert_t,
        train_size=0.999,
        random_seed=random_seed,
        root_dir=rootDir,
        pathFileValid=pathFileValid,
        pathFileTrain=pathFileTrain,
    )
    
    # Check CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    model_defer = DenseNet121_defer(nnClassCount).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_defer = torch.nn.DataParallel(model_defer)
    batch, losst, losse = CheXpertTrainer_defer.train_defer(
            model_defer,
            rad_index=1,
            learn_to_defer=False,
            dataLoadertrain=dataLoaderTrain_t,
            dataLoaderVal=dataLoaderTest_t,
            nnClassCount=nnClassCount,
            trMaxEpoch=3,
            launchTimestamp=timestampLaunch,
            alpha=1*[alpha],
            checkpoint=pretrained_model
    )
    torch.save(model_defer.state_dict(), pretrained_model)

    base_path = os.path.join(
        script_dir, "checkpoints", "perstep-rerun-curve", timestampLaunch, str(random_seed)
    )
    os.makedirs(base_path, exist_ok=True)
    dense_per_step_model = f"{base_path}/densenet_defer_step_{t}.pth"
    os.makedirs(base_path, exist_ok=True)

    batch, losst, losse = CheXpertTrainer_defer.train_defer(
            model_defer,
            rad_index=1,
            learn_to_defer=True,
            dataLoadertrain=dataLoaderTrain_t,
            dataLoaderVal=dataLoaderTest_t,
            nnClassCount=nnClassCount,
            trMaxEpoch=1,
            launchTimestamp=timestampLaunch,
            alpha=alpha,
            checkpoint=dense_per_step_model
        )
    loss, auc_cls_per_class, auc_exp_per_class, auc_sys_per_class, defer_rates, auprc_cls_per_class, auprc_exp_per_class, auprc_sys_per_class = trainer_t.test_epoch_defer(model_defer, dataLoaderTest_t, alpha, DEVICE, rad_index=1, use_defer=True)

    for i in range(14):
        new_row = pd.DataFrame({
            'Timestep': [t],
            'Class': [i],
            'AUC_cls': [auc_cls_per_class[i]],
            'AUC_exp': [auc_exp_per_class[i]],
            'AUC_sys': [auc_sys_per_class[i]],
            'Defer Rate': [defer_rates[i]],
            'AUPRC_cls': [auprc_cls_per_class[i]],
            'AUPRC_exp': [auprc_exp_per_class[i]],
            'AUPRC_sys': [auprc_sys_per_class[i]]
        })
        df = pd.concat([df, new_row], ignore_index=True)

    base_path = os.path.join(
        script_dir, "results", "perstep-rerun-curve", timestampLaunch, str(random_seed)
    )
    os.makedirs(base_path, exist_ok=True)
    df.to_csv(f"{base_path}/perstep-full.csv", index=False)
    torch.save(model_defer.state_dict(), dense_per_step_model)