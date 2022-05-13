# Text Language Predictor

The tool has the following functionality:
* languages prediction
* span extraction for every language

```
> Мечта может стать support or source leiden. Mai träumen insanı həyatla doldurmaq немесе оны өлтіріңіз.
ru: Мечта может стать
en: support or source
de: leiden Mai träumen
az: insanı həyatla doldurmaq
kk: немесе оны өлтіріңіз
```

## Dependencies
To install all the dependencies run from the project root:
```
pip install -r requirements.txt
```

## Training
The training is written in PyTorch using PyTorch Lightning framework. Logging is done with 
W&B, you can see runs [here](https://wandb.ai/falca/text-lang-predictor?workspace=user-falca).

All the model and dataset details can be found in:
* [`MultiLanguageDataset`](https://github.com/Mogreine/text-language-predictor/blob/main/src/training/dataset.py#L35)
* [`BertLangNER`](https://github.com/Mogreine/text-language-predictor/blob/main/src/training/model.py#L11)

respectively.

### Data
#### Raw data source
Since there is no actual dataset for such a task the only option left is to generate synthetic dataset. 
There are many multilingual dataset, some of them:
* OpenSubtitles -- consists of subtitles to movies. Quite a big and clean dataset. But some required languages are missing.
* MC4 -- a multilingual colossal, cleaned version of Common Crawl's web crawl corpus. Also quite a big dataset, but as clean as the previous one (e.g. russian texts may contain english words). Hebrew language is missing.
* WikiAnn -- a multilingual named entity recognition dataset consisting of Wikipedia articles. A small dataset with ~20k examples almost for every required language. Data is not so clean but all required languages are present.

I decided to choose WikiAnn dataset, because:
1. Easy to use -- it has all the required languages so there is no need to merge multiple datasets.
2. Balanced and not big -- almost every language has 20k samples.

#### Sampling strategy
For now, we have texts and corresponding languages. But we cannot use it as samples, because then there will be only one language per sample.
Let's create samples by mixing texts from different languages.

First we have to choose a number of different texts (sentences) `n` in a sample. I have tested 3 strategies:
1. Constant -- every sample consists of fixed number of sentences
2. Uniform -- every sample consists of `n` sentences. Where `n` is uniformly distributed between 1 and `n_sentences`
3. Custom -- the number of sentences is sampled from the defined number of `lengths` with defined probabilities `length_probs`

Once number of sentences `n` is defined then
1. Sample `n` languages (each with equal probability)
2. Sample random texts
3. Shuffle texts
4. Concatenate texts into sample

Also, with `word_perm_prob` probability all the words in a sample are shuffled.

### Usage

To run training run from the root:
```
PYTHONPATH=. python src/training/train.py
```

All the training parameters are stored in `src/configs/bert_config.yaml`. To see the description
 you can run:
```
PYTHONPATH=. python src/training/train.py --help
```
You can set parameters either by editing the config file or directly through command-line.

## Inference
You can play with the model using `parse.py` script:
```
PYTHONPATH=. python src/inference/parse.py
```

To see script parameters, run:
```
PYTHONPATH=. python src/inference/parse.py --help
```

## Tests
To run marker tests for the model, run:
```
PYTHONPATH=. pytest tests/model_tests.py
```
