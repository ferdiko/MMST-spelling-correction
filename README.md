# MMST spelling correction on Twitter data
[![Build Status](https://travis-ci.com/ferdiko/MMST-spelling-correction.svg?branch=master)](https://travis-ci.com/ferdiko/MMST-spelling-correction)

This repository contains a basic python implementation of the minial minimum spanning tree (MMST) spelling corrector. MMST is a novel context sensitive spelling corrector that exploits clustering of similar words in GloVe embeddings. This enables it to decide on the right word for correction given several candidates.

The implementation of the MMST algorithm is contained in ```embed/mmst.py```. A multithreaded manager class, that applies the correction to a text file is contained in ```preprocessing/spelling_correction_mmst.py```, a pipeline applying MMST correction and further performance enhancing preprocessing is contained in ```preprocessing/pipeline_mmst.py```.

The project ensued ETH Zurich's Computational Intelligence Lab 2020, in which we were given the task to perform sentiment analysis on Twitter data. We hereby came up with MMST correction since Tweets have many misspellings which often are non-trivial to correct. We furthermore scraped slang and abbreviation dictionaries, such that correctly spelled slang/abbreviations is not falsely corrected to standard English. The repository also contains several models we used for classification.

For a more detailed description of the methods and the results, please refer to [our report](report.pdf).

MMST spelling correction was evaluated in terms of classification accuracy when running a model on data corrected with MMST versus baselines. It hereby led to an improved accuracy compared to all spelling correction libraries we benchmarked, as well as to no correction.

## Installation

For smooth running of the preprocessing and models we use pipenv library to manage virtual environments and dependencies on different devices. You can install via

1. Install `pipenv` with
```
pip install pipenv
```
2. Spin up a virtual environment with:
```
pipenv shell
```
3. Install dependencies (`--dev` option needed for some of the preprocessing steps)
```
pipenv install
```

## Structure

We divided the project into multiple folders corresponding to steps in the whole ml pipeline:

```
  .
  ├── data/                   # raw/processed datasets and data loaders
  ├── embed/                  # vocabularies and embeddings/vectorizers
  ├── model/                  # models and interceptors for training
  ├── preprocessing/          # pipelines and helpers for preprocessing data
  ├── run_predict.py          # script for loading saved models and predicting on test data
  ├── run_preprocessing.py    # script for executing preprocessing on data
  ├── run_training.py         # script for training model with TensorFlow
  ├── run_training_pytorch.py # script for training model with PyTorch
  └── README.md
```

### Script `run_preprocessing.py`

The preprocessing script is a playground where you can build up a complete preprocessing pipeline with the helper functions from the `/preprocessing` directory. All transformers follow the same interface, which simplifies chaining preprocessing steps.

See [Preproccessing](../preprocessing/README.md) for more information

### Script `run_training.py`

The training script loads the training data and a specific model from the `/model` directory. All models inherit the same base class to provide a consistent experience for this script. The models need to implement `build` and `fit` functions that take similar base parameters.

See [Model](../model/README.md) for more information

### Script `run_training_pytorch.py`

The training script loads the training data and a specific model from the `/model` directory with the PyTorch framework. All models inherit the same base class to provide a consistent experience for this script.

See [Model](../model/README.md) for more information

### Script `run_predict.py`

This script loads the saved models from the training and predicts classification responses from test data.

## Datasets
The unprocessed data can be found [here](https://www.kaggle.com/c/cil-text-classification-2020).

For convinience, we also provide links to the already processed data sets as downloadable zip folders.
These links are included in text files in ```data```. Further description of the processing applied can be found in the [report](report.pdf) or the [readme](preprocessing/README.md) of ```preprocessing```.
