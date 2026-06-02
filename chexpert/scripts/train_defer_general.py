import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.densenet_defer_seq import DenseNet121SeqDefer, SeqTrainerDefer
from data.cheXpert_bias import split_dataset_seq
from experts.fake_bias_curve import ExpertModelCurveBiased
import random
import numpy as np

nnClassCount = 14
seq_len      = 21
step         = 10
batch_size   = 8

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_data = os.path.join(script_dir, "datasets")
rootDir = os.environ.get("CHEXPERT_DATA_ROOT", _default_data)
pathFileTrain = os.path.join(rootDir, "CheXpert-v1.0-small", "train.csv")
pathFileValid = os.path.join(rootDir, "CheXpert-v1.0-small", "valid.csv")

BACKBONE_CKPT       = os.path.join(script_dir, "checkpoints",
                                   "pretrained_step_model_full_pretrained_no_alpha.pth")
GENERAL_PRETRAIN_CKPT = os.path.join(script_dir, "checkpoints",
                                     "general_curve_21steps", "pretrained_general.pth")

seeds = [42, 43, 44, 45, 46]


def load_backbone_from_perstep(model, ckpt_path):
    """Remap DenseNet121_defer keys to DenseNet121SeqDefer backbone keys."""
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Backbone checkpoint not found at {ckpt_path}; using ImageNet weights.")
        return
    src = torch.load(ckpt_path, map_location="cpu")
    if isinstance(src, dict) and "state_dict" in src:
        src = src["state_dict"]
    dst      = model.state_dict()
    remapped = {}
    for k, v in src.items():
        k = k.replace("module.", "")
        if k.startswith("densenet121.features."):
            new_k = k.replace("densenet121.features.", "backbone.features.")
            if new_k in dst and dst[new_k].shape == v.shape:
                remapped[new_k] = v
    dst.update(remapped)
    model.load_state_dict(dst)
    print(f"[INFO] Loaded {len(remapped)} backbone layer(s) from {ckpt_path}")


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

# Phase 1: pretrain one shared general model (no deferral), reuse for all seeds.
if not os.path.exists(GENERAL_PRETRAIN_CKPT):
    print("\n[Phase 1] Pretraining general model (3 epochs, no deferral)...")
    set_seed(42)

    exp_pre = ExpertModelCurveBiased(
        confounding_class=13, seq_len=seq_len, num_classes=nnClassCount
    )
    loaderTrain_pre, loaderVal_pre, _ = split_dataset_seq(
        train_size=0.999, random_seed=66,
        root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain,
        exp_fake=exp_pre, trBatchSize=batch_size, seq_len=seq_len, step=step,
    )

    model_pre = DenseNet121SeqDefer(num_classes=nnClassCount).to(device)
    load_backbone_from_perstep(model_pre, BACKBONE_CKPT)

    trainer_pre = SeqTrainerDefer(
        model_pre, exp_pre, device, seed="pretrain",
        result_dir="results/general_curve_21steps",
    )
    trainer_pre.fit(loaderTrain_pre, loaderVal_pre,
                    pretained_epochs=3, finetuned_epochs=0, lr=1e-4)

    os.makedirs(os.path.dirname(GENERAL_PRETRAIN_CKPT), exist_ok=True)
    torch.save(model_pre.state_dict(), GENERAL_PRETRAIN_CKPT)
    print(f"[Phase 1] Saved shared pretrained model -> {GENERAL_PRETRAIN_CKPT}")

    del model_pre, trainer_pre, loaderTrain_pre, loaderVal_pre
    torch.cuda.empty_cache()
else:
    print(f"[Phase 1] Reusing existing pretrained model: {GENERAL_PRETRAIN_CKPT}")

# Phase 2: finetune with deferral per seed from shared Phase-1 checkpoint.
for seed in seeds:
    print(f"\n[Phase 2] Seed {seed}: loading pretrained general -> 1 defer epoch...")
    set_seed(seed)

    exp_biased = ExpertModelCurveBiased(
        confounding_class=13, seq_len=seq_len, num_classes=nnClassCount
    )
    loaderTrain, loaderVal, loaderTest = split_dataset_seq(
        train_size=0.999, random_seed=66,
        root_dir=rootDir, pathFileValid=pathFileValid, pathFileTrain=pathFileTrain,
        exp_fake=exp_biased, trBatchSize=batch_size, seq_len=seq_len, step=step,
    )

    model = DenseNet121SeqDefer(num_classes=nnClassCount).to(device)
    model.load_state_dict(torch.load(GENERAL_PRETRAIN_CKPT, map_location=device))

    trainer = SeqTrainerDefer(
        model, exp_biased, device, seed=seed,
        result_dir="results/general_curve_21steps",
    )
    trainer.fit(loaderTrain, loaderTest, pretained_epochs=0, finetuned_epochs=1, lr=1e-4)

    ckpt_dir = os.path.join("checkpoints", "general_curve_21steps", str(seed))
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(ckpt_dir, "densenet_seq_defer_curve_21steps.pth"))
    print(f"Seed {seed} done -> {ckpt_dir}")

    del model, trainer
    torch.cuda.empty_cache()
