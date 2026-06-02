import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.densenet_lstm import DenseLSTMDefer, CheXpertTrainerDeferLSTM
from data.cheXpert_bias import split_dataset_seq
from experts.fake_bias_curve import ExpertModelCurveBiased
import random
import numpy as np

parser = argparse.ArgumentParser(description="Train DenseLSTM Defer — two-phase approach")
parser.add_argument("--hidden_dim",        type=int,   default=1024)
parser.add_argument("--lr",                type=float, default=0.001)
parser.add_argument("--batch_size",        type=int,   default=16)
parser.add_argument("--seq_len",           type=int,   default=21)
parser.add_argument("--step",              type=int,   default=10)
parser.add_argument("--lstm_layers",       type=int,   default=1)
parser.add_argument("--train_size",        type=float, default=0.999)
parser.add_argument("--random_seed",       type=int,   default=66)
parser.add_argument("--pretrained_epochs", type=int,   default=3)
parser.add_argument("--finetuned_epochs",  type=int,   default=4)
parser.add_argument("--checkpoint_dir",    type=str,   default="checkpoints/lstm_curve_21steps")
parser.add_argument("--model_name",        type=str,   default="densenet_lstm_21steps")
parser.add_argument("--result_dir",        type=str,   default="results/lstm_curve_21steps")
args = parser.parse_args()
print(args)

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(args.checkpoint_dir):
    args.checkpoint_dir = os.path.join(script_dir, args.checkpoint_dir)
if not os.path.isabs(args.result_dir):
    args.result_dir = os.path.join(script_dir, args.result_dir)

nnClassCount = 14
seq_len      = args.seq_len
step         = args.step
batch_size   = args.batch_size
hidden_dim   = args.hidden_dim

_default_data = os.path.join(script_dir, "datasets")
rootDir = os.environ.get("CHEXPERT_DATA_ROOT", _default_data)
pathFileTrain = os.path.join(rootDir, "CheXpert-v1.0-small", "train.csv")
pathFileValid = os.path.join(rootDir, "CheXpert-v1.0-small", "valid.csv")

LSTM_PRETRAIN_CKPT = os.path.join(args.checkpoint_dir, "pretrained_lstm.pth")

seeds = [42, 43, 44, 45, 46]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

exp_biased = ExpertModelCurveBiased(
    confounding_class=13, seq_len=seq_len, num_classes=nnClassCount
)

# Phase 1: pretrain one shared LSTM (no deferral), reused for all seeds.
if not os.path.exists(LSTM_PRETRAIN_CKPT):
    print(f"\n[Phase 1] Pretraining LSTM ({args.pretrained_epochs} epochs, no deferral)...")
    set_seed(42)

    loaderTrain_pre, loaderVal_pre, loaderTest_pre = split_dataset_seq(
        train_size=args.train_size, random_seed=args.random_seed,
        root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain,
        exp_fake=exp_biased, trBatchSize=batch_size, seq_len=seq_len, step=step,
    )

    model_pre = DenseLSTMDefer(
        num_classes=nnClassCount,
        lstm_hidden=hidden_dim,
        lstm_layers=args.lstm_layers,
    ).to(device)

    trainer_pre = CheXpertTrainerDeferLSTM(
        model_pre, exp_biased, device,
        lr=args.lr,
        pretrained_epochs=args.pretrained_epochs,
        finetuned_epochs=0,           # pretrain only
        seq_len=seq_len, step=step,
        result_dir=args.result_dir,
    )
    trainer_pre.train_defer_lstm(loaderTrain_pre, loaderTest_pre, seed=42)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(model_pre.state_dict(), LSTM_PRETRAIN_CKPT)
    print(f"[Phase 1] Saved shared LSTM pretrained model -> {LSTM_PRETRAIN_CKPT}")

    del model_pre, trainer_pre, loaderTrain_pre, loaderVal_pre, loaderTest_pre
    torch.cuda.empty_cache()
else:
    print(f"[Phase 1] Reusing existing pretrained model: {LSTM_PRETRAIN_CKPT}")

# Phase 2: finetune with deferral per seed from shared Phase-1 checkpoint.
for seed in seeds:
    print(f"\n[Phase 2] Seed {seed}: loading shared pretrained -> {args.finetuned_epochs} defer epochs...")
    set_seed(seed)

    loaderTrain, loaderVal, loaderTest = split_dataset_seq(
        train_size=args.train_size, random_seed=args.random_seed,
        root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain,
        exp_fake=exp_biased, trBatchSize=batch_size, seq_len=seq_len, step=step,
    )

    model = DenseLSTMDefer(
        num_classes=nnClassCount,
        lstm_hidden=hidden_dim,
        lstm_layers=args.lstm_layers,
    ).to(device)
    model.load_state_dict(torch.load(LSTM_PRETRAIN_CKPT, map_location=device))

    trainer = CheXpertTrainerDeferLSTM(
        model, exp_biased, device,
        lr=args.lr,
        pretrained_epochs=0,          # skip pretrain; model already loaded
        finetuned_epochs=args.finetuned_epochs,
        seq_len=seq_len, step=step,
        result_dir=args.result_dir,
    )
    trainer.train_defer_lstm(loaderTrain, loaderTest, seed=seed)

    ckpt_dir = os.path.join(args.checkpoint_dir, str(seed))
    os.makedirs(ckpt_dir, exist_ok=True)
    save_name = (f"{args.model_name}-{hidden_dim}-unit-{args.lstm_layers}-layers-"
                 f"{args.pretrained_epochs}-pretrained-{args.finetuned_epochs}-finetuned-"
                 f"no-alpha-{seq_len}-steps-curve.pth")
    torch.save(model.state_dict(), os.path.join(ckpt_dir, save_name))
    print(f"Seed {seed} done -> {ckpt_dir}/{save_name}")

    del model, trainer
    torch.cuda.empty_cache()
