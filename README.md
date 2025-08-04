# Claim Denial Classification

This repository contains a small Python package and command–line interface
for training, evaluating and running inference on a binary insurance
claim denial classification task.  A dataset of insurance claims and
denial labels is downloaded from the Hugging Face Hub or can be supplied
as a local CSV file.  Models are trained by fine‑tuning a pretrained
Transformer, such as [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased),
using the [Hugging Face `Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) API.

## Installation

To install the package and its dependencies, clone this repository and run:

```bash
pip install -r requirements.txt
pip install -e .
```

This will expose a console script called `claim-denial` on your system.

## Usage

The CLI supports three subcommands: `train`, `evaluate` and `infer`.

### Train

Fine‑tune a pretrained model on the claim denial dataset:

```bash
claim-denial train \
  --model_name distilbert-base-uncased \
  --output_dir outputs \
  --epochs 3
```

You can optionally provide your own CSV file containing two columns:
`claim_text` and `denial_label`.

### Evaluate

Evaluate a saved model directory on the default test split or a custom
CSV:

```bash
claim-denial evaluate --model_dir outputs/claim_denial_model
```

### Infer

Run inference on a single input string:

```bash
claim-denial infer --model_dir outputs/claim_denial_model "The insurer has denied my claim due to late payment."
```

## Project Structure

```
claim_denial/
├── cli/          # Command‑line interface definitions
├── data/         # Functions for loading and splitting data
├── model/        # Training, evaluation, baseline and inference utilities
└── __init__.py   # Top‑level package initialisation
```

The package is designed to be modular; individual functions can be
imported and used directly from Python.  See the docstrings in the
respective modules for more details.

## Notes

* Training large models requires a GPU and sufficient memory.  Adjust the
  batch size and number of epochs according to your hardware.
* When a local CSV file is provided, its columns are automatically
  detected; the code expects at least `claim_text` and `denial_label`.
