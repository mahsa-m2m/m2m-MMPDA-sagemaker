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
To do inference, run `model_inference/main.py`. It gets single/batch videos.

Set the video path in => `main.py`

Set the model path and config in => `config.py`
```
python model_inference/main.py
```
### 🔎🧠 5. Explainability
Generate and save heatmaps on the video frames to show what features the model has learned.

Set the video path in => `visualize_qualitative.py`

Set the model path and config in => `explainer.py`
```
python model_explain/visualize_qualitative.py
```
The output is saved as `explainability_result.avi`. 
