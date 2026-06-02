Training code for three setups: CIFAR-10 (image), CheXpert (chest X-ray), and hate-speech text.
Plotting utilities are not included.

Layout
------
cifar10/
  common/          model and training helpers
  expert/          place expert accuracy CSVs here (see env vars)
  scripts/         cifar10_defer_general.py, cifar10_defer_perstep.py, cifar10_defer_lstm_t.py

chexpert/
  models/, data/, experts/, utils/
  scripts/         train_defer_general.py, train_defer_perstep.py, train_defer_lstm.py
  checkpoints/     create and place backbone or pretrained weights as needed
  datasets/        default parent for CheXpert-v1.0-small (or set CHEXPERT_DATA_ROOT)

hatespeech/
  scripts/         hatespeech_defer_general.py, hatespeech_defer_perstep.py, hatespeech_defer_lstm.py
  data/            labeled CSV for torchtext TabularDataset
  expert/          expert curve CSV (one row per timestep)
  third_party/twitteraae/model/  TwitterAAE vocab and count table files (or set env vars)

Environment variables (optional)
--------------------------------
CIFAR10_EXPERT_CURVE       path to expert curve CSV (default: cifar10/expert/curve_50.csv;
                           LSTM script defaults to expert/curve_new_85.csv when unset)
CIFAR10_BACKBONE_CKPT      WideResNet backbone for LSTM script (default under cifar10/models/)

CHEXPERT_DATA_ROOT         directory containing CheXpert-v1.0-small/train.csv and valid.csv
CHEXPERT_PRETRAINED_STEP   DenseNet checkpoint for per-step training

HATESPEECH_LABELED_CSV     labeled training CSV path
HATESPEECH_EXPERT_CURVE    expert timestep curve CSV
TWITTERAAE_VOCAB           TwitterAAE model_vocab.txt
TWITTERAAE_COUNTS          TwitterAAE model_count_table.txt

How to run (from repository root)
---------------------------------
Install dependencies: pip install -r training_code_release/requirements.txt

CIFAR-10 (run with working directory = training_code_release/cifar10):
  python scripts/cifar10_defer_general.py
  python scripts/cifar10_defer_perstep.py
  python scripts/cifar10_defer_lstm_t.py

CheXpert (working directory = training_code_release/chexpert):
  python scripts/train_defer_general.py
  python scripts/train_defer_perstep.py --seed 66
  python scripts/train_defer_lstm.py

Hate speech (working directory = training_code_release/hatespeech):
  python scripts/hatespeech_defer_general.py
  python scripts/hatespeech_defer_perstep.py
  python scripts/hatespeech_defer_lstm.py

Notes
-----
- CIFAR-10 uses torchvision CIFAR10 download/cache.
- CheXpert scripts expect ImageNet-pretrained DenseNet where applicable.
- Hate-speech scripts use torchtext 0.4-style APIs and spaCy tokenization; install en_core_web_sm if required by your torchtext/spaCy version.
