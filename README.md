# Text Language Predictor

The tool has the following functionality:
* languages prediction
* span extraction for every language

## Examples of usage

## Training
The training is written in PyTorch using PyTorch Lightning framework. Logging is done with 
W&B, you can see runs [here](https://wandb.ai/falca/text-lang-predictor?workspace=user-falca).

All the model and dataset details can be found in:
* `MultiLanguageDataset`
* `BertLangNER`

respectively.

### Data
Describe the dataset

### Usage

To run training run from the root:
```
PYOTHNONPATH=. python src/training/train.py
```

All the training parameters are stored in `src/configs/bert_config.yaml`. To see the description
 you can run:
```
PYOTHNONPATH=. python src/training/train.py --help
```
You can set parameters either by editing the config file or directly through command-line.

# Inference
