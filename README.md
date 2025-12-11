## Prodigy Intelligence Multi-Modal Deception Detection

This repository contains code for training a multi-modal model to detect deception in a video dataset.

This repository was cloned from here

https://github.com/RH-Lin/MMPDA


## Reproduce Steps

### 🛠️ 1. Installation:

```bash
pip install -r requirements.txt
```

### 📊 2. Create dataset csv file (train, val, test)

```bash
python data_util/csv_creator.py
```

### 🚀 3. Train and eval
To start the training, run `train_test_features.py` with the appropriate parameters. For example:

```bash
python train_test_feature.py --train_list sample/train.csv --val_list sample/validation.csv --batchsize 4 --max_epochs 10 --fusion_type mult
```

### 💡 4. Inference
(Will be added)
