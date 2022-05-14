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

## Experiments
### Data
#### Raw data source
Since there is no actual dataset for such a task the only option left is to generate synthetic dataset. 
There are many multilingual dataset, some of them:
* OpenSubtitles -- consists of subtitles to movies. Quite a big and clean dataset. But some required languages are missing.
* MC4 -- a multilingual colossal, cleaned version of Common Crawl's web crawl corpus. Also quite a big dataset, but as clean as the previous one (e.g. russian texts may contain english words). Hebrew language is missing.
* WikiAnn -- a multilingual named entity recognition dataset consisting of Wikipedia articles. A small dataset with ~20k examples almost for every required language. Data is not so clean but all required languages are present.

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

### Model
I used pretrained BERT (bert-base-multilingual-cased). Didn't really have a choice here since there is not 
so many models pretrained for all the required languages. 

#### Thoughts and process
First, I decided to choose WikiAnn dataset, because:
1. Easy to use -- it has all the required languages so there is no need to merge multiple datasets.
2. Balanced and not big -- almost every language has 20k samples.

The results were mostly ok, but the data is really dirty -- many languages have sentences with only english words. 
And when the model sees sentences with some english words in it, it tends not to notice english language.

I started to experiment with sentence sampling strategies so that the model saw not only very long examples (512 tokens) 
but also small ones. And uniform distribution of lengths seems to be the best method (among described). 

Since wikiann seems to mix languages, I decided to try using OpenSubtitles dataset because it was way cleaner. 
The problem is that, first, 'az' and 'be' languages are missing. And secondly, 
it is too big for a task -- some languages have more than 3 GB of raw data. So I decided to do the following:
* crop existing languages' data to ~150 MB (at most)
* take missing languages from MC4 dataset

Everything would be fine but MC4 data differs form OpenSubtitles -- samples are way longer, and they also have other languages it them (mostly en).
 So I split samples into sentences and then took a random one. The sentence were also truncated if had >15 words. 
It was necessary to do because otherwise long russian texts were considered to be belarusian by the model.

#### TL;DR
![](data/meme.jpg)
## Tests
To run marker tests for the model, run:
```
PYTHONPATH=. pytest tests/model_tests.py
```

## Further work
1. Using cleaner data for az and be languages might improve the quality. 
2. Using better techniques for NER (e.g. [here](https://arxiv.org/pdf/1910.11476v6.pdf)). We could first find spans and then classify them.
3. Train model and tokenizer for required languages specifically. In this case vocabulary will be way more suitable.
4. May try using character level transformers -- to understand the language of a word it might be enough to look only at symbols.
5. Bert is trained with absolute positional embeddings. Though to understand to language of a token it is more useful to have the information about the nearest tokens directly. So using relative positional embeddings seems to be more suitable.