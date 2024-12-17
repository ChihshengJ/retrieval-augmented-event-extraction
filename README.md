# Retrieval Augmented Event Extraction

This repository contains the data and source code for the Retrieval Augmented Event project.

This project used the [FAMuS][https://github.com/FACTSlab/FAMuS/tree/main] dataset as the source for event frames and training data. The input requires a recognized event documented in the [FrameNet][https://framenet.icsi.berkeley.edu/frameIndex] ontology and the document itself. Since the encoder is based on Longformer, the maximum sequence length is clipped at 1024 tokens (which could translate to 700-900 words in English), which is considerably long for tasks like this but 



## Python Environment Setup

We recommend using a fresh Python environment for training/evaluating our model scripts on the dataset. We tested our codebase on Python 3.8

You can create a new conda environment by running the following command:

```
conda create -n RAEE python=3.8
conda activate RAEE
```

Before installing requirements, we recommend you install PyTorch on your system based on your GPU/CPU configurations following the instructions [here](https://pytorch.org/get-started/locally/).
Then inside the root directory of this repo, run:

```
pip install -r requirements.txt
```

## Training Dataset setup

You can train the encoder locally by directly using the dataset in this repo. However, the training set is tokenized by Longformer's tokenizer, and LLM's prediction used for training was collected from GPT-4o's output. The module supports using other pre-trained models and LLM APIs to do the training data collection, it would roughly take 5 hours to collect the predictions from an LLM similar to GPT-4o and the training would take about 1 hour on a single GPU.

## Usage

### Inference with current checkpoint

### Using other LLM APIs for inference



## Evaluation Metric - CEAF_RME (a)

The CEAF_RME(a) metric that uses the normalized edit distance (read more in the paper) is implemented in the Iter-X repo and we provide an [example notebook](https://github.com/sidsvash26/iterx/blob/sv/famus/src/iterx/metrics/famus/metric_example_notebook.ipynb) showing how to compute the metric on a sample gold and predicted template set from a document.
