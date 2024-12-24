# English-Vietnamese Neural Machine Translation

This project implements an English-Vietnamese neural machine translation (NMT) model based on the **"Attention is All You Need"** paper. The model is built from scratch using PyTorch and trained on the [English-Vietnamese Translation dataset](https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation) from Kaggle.

## Model Architecture

The model follows the Transformer architecture described in the original paper, consisting of:

-   **Encoder:** A stack of encoder layers, each containing:
    -   Multi-head self-attention mechanism
    -   Position-wise feedforward network
    -   Residual connections and layer normalization
-   **Decoder:** A stack of decoder layers, each containing:
    -   Masked multi-head self-attention mechanism
    -   Multi-head attention mechanism over the encoder output
    -   Position-wise feedforward network
    -   Residual connections and layer normalization
-   **Positional Embeddings:** Combines word embeddings with positional encodings to provide information about the order of words in the sequence.

## Performance

The model achieves a **test loss of 0.4162** using **cross-entropy loss**.

## Prerequisites

-   Python 3.x
-   PyTorch
-   SentencePiece
-   NumPy

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. **Install the required packages:**
    ```bash
    pip install torch sentencepiece numpy
    ```
3. **Download the pre-trained model and SentencePiece model:**
    - Download `translator_model.pth` and `sentencePiece_model.model` from the `model` directory and place them in the `model` folder.

## Usage

To translate an English sentence into Vietnamese, run the `run.py` script with the input sentence as a command-line argument:

```bash
python3 run.py "Your English sentence here."
```

## Dataset

The model is trained on the English-Vietnamese Translation dataset available on Kaggle.

## Implementation Details

-   From Scratch: The entire model is implemented from scratch in PyTorch, including all components of the Transformer architecture.

-   Greedy Decoding: The run.py script uses greedy decoding to generate the translated output sequence.

-   SentencePiece Tokenization: SentencePiece is used for subword tokenization of both English and Vietnamese sentences.
