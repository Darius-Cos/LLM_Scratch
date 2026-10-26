# LLM From Scratch — Building a Character-Level GPT with PyTorch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
![Project](https://img.shields.io/badge/Project-Educational-blue)

An educational deep learning project exploring **how language models work internally** by building two character-level text generators in PyTorch: a simple **Bigram Language Model** and a small **decoder-only GPT Transformer**.

The models learn from Shakespeare text and generate new characters one at a time. The notebooks emphasize understanding the full pipeline—from tokenization to attention, backpropagation, training, and autoregressive text generation—rather than using a pretrained language model.

> **Scope:** This is a learning-oriented implementation of a GPT-style architecture, not a production LLM or a model trained on internet-scale data. The neural-network building blocks are implemented with PyTorch.

## Project overview

| Notebook | Model | Main concepts |
| --- | --- | --- |
| [LLM_From_Scractch_var1.ipynb](LLM_From_Scractch_var1.ipynb) | Bigram baseline | Character tokenization, embeddings as lookup tables, logits, softmax, cross-entropy, backpropagation, AdamW, sampling |
| [LLM_Scractch_Var_2.ipynb](LLM_Scractch_Var_2.ipynb) | GPT-style Transformer | Token and position embeddings, causal self-attention, multi-head attention, feed-forward networks, residual connections, LayerNorm, training and generation |

**Dataset and outputs**

- [`input.txt`](input.txt) — Shakespeare training text, approximately 1.1 million characters.
- [`more.txt`](more.txt) — a saved generated-text sample.

## Architecture

The second notebook implements a **character-level, autoregressive, decoder-only Transformer**:

```text
Shakespeare text
       |
       v
Character tokenizer (character <-> integer ID)
       |
       v
Token embeddings + learned positional embeddings
       |
       v
+---------------------------------------------+
| Transformer Block                         x6 |
|                                             |
|   LayerNorm                                 |
|       |                                     |
|   Multi-Head Causal Self-Attention (6 heads)|
|       |                                     |
|   Residual connection                       |
|       |                                     |
|   LayerNorm                                 |
|       |                                     |
|   Feed-Forward Network (Linear/ReLU/Linear) |
|       |                                     |
|   Residual connection                       |
+---------------------------------------------+
       |
       v
Final LayerNorm
       |
       v
Linear language-model head
       |
       v
Logits -> Softmax -> Sample next character
       |
       +---- Append character and repeat
```

Each attention head produces **Query (Q), Key (K), and Value (V)** vectors. The model computes scaled dot-product attention:

```text
Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k) + causal_mask) @ V
```

The causal mask prevents tokens from attending to future positions. Multiple heads learn complementary relationships, while feed-forward layers transform each token's representation.

## Model configuration

The GPT notebook uses the following hyperparameters:

| Hyperparameter | Value |
| --- | ---: |
| Vocabulary size | 65 characters |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Context length | 256 characters |
| Batch size | 64 |
| Dropout | 0.2 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Training steps | 5,000 |
| Training / validation split | 90% / 10% |
| Total GPT parameters | **10,788,929 (~10.79M)** |

The Bigram baseline uses a `65 x 65` lookup table (**4,225 parameters**) to predict the next character using only the current character. Unlike the Transformer, it does not model a longer context.

## Training results

The following numbers are **recorded outputs from the GPT notebook**, not benchmark averages or guaranteed reproduction results:

| Step | Training loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.1626 | 4.1595 |
| 500 | 1.7616 | 1.9170 |
| 1,000 | 1.3982 | 1.6055 |
| 2,000 | 1.1904 | 1.4882 |
| 3,000 | 1.0756 | **1.4793** |
| 4,000 | 0.9675 | 1.5022 |
| 4,999 | 0.8624 | 1.5650 |

Validation loss improves substantially during training, then starts increasing while training loss continues to decrease—a useful example of **overfitting**. In this run, the best recorded validation loss is around step 3,000; the notebook does not implement early stopping or automatic checkpoint selection.

### Generated text example

A short excerpt from the GPT notebook's saved output:

```text
O heavy heavy blood hat, I know, our leleve.

KING RICHAR II:
You well?

TRANIO:
Bid me some senses of jumpet the queen your mind,
Your honour tell me what you are.
```

The output imitates aspects of the Shakespeare dataset's formatting and vocabulary, but it contains invented words and inconsistent meaning. This is expected for a small character-level model trained on a limited dataset.

## Run in Google Colab

The notebooks were written for **Google Colab** and use Google Drive to locate the training file.

1. Open either notebook:
   - [Bigram — open in Colab](https://colab.research.google.com/github/Darius-Cos/LLM_Scractch/blob/main/LLM_From_Scractch_var1.ipynb)
   - [GPT Transformer — open in Colab](https://colab.research.google.com/github/Darius-Cos/LLM_Scractch/blob/main/LLM_Scractch_Var_2.ipynb)
2. In Colab, choose **Runtime → Change runtime type → T4 GPU** if available, especially for the GPT notebook.
3. Mount Google Drive when prompted by the first cell.
4. Put [`input.txt`](input.txt) in the folder configured in the notebook, or change the dataset path to your own location. The notebooks currently reference:

   ```python
   /content/drive/MyDrive/03_Proiecte/LLM_Scratch/input.txt
   ```

5. Run the cells **top to bottom**. The GPT notebook trains its model and generates a text sample in the last cell.

Alternatively, download `input.txt` into Colab and adjust the notebook's file-loading cell to use the local `input.txt` path.

**Requirements:** Python, PyTorch, and Google Colab (or a compatible Jupyter environment). GPU acceleration is recommended for the larger Transformer.

## What I learned

This project was built as a step-by-step exploration of:

- **Tokenization:** mapping individual characters to IDs and back using `stoi`, `itos`, encoders and decoders.
- **Embeddings:** replacing discrete token IDs with trainable vectors.
- **Training examples:** shifting character sequences to create input/target pairs.
- **Language modeling:** logits, softmax probabilities and cross-entropy loss.
- **Self-attention:** constructing Q/K/V projections, attention scores, a causal mask and weighted value aggregation.
- **Transformer design:** combining attention heads, feed-forward layers, LayerNorm and residual connections.
- **Optimization:** calculating gradients with `loss.backward()` and updating weights with AdamW.
- **Text generation:** repeatedly sampling a new character from the model's predicted distribution.
- **Evaluation:** comparing training and validation losses to recognize overfitting.

## Implementation notes and limitations

- **Educational character tokenizer:** the GPT notebook currently builds its vocabulary with `list(set(text))`, so the character-to-ID mapping can change across fresh runs. A sorted vocabulary and saving the mapping alongside checkpoints would improve reproducibility.
- **Bigram notebook split-name mismatch:** its `get_batch()` checks for `'train_data'`, while some calls pass `'train'`. In the current version, this can accidentally select the validation split. Change the condition to `if split == 'train':` before interpreting its training results.
- **Colab-specific file paths:** update the Google Drive path to run the notebooks under another account or locally.
- **No production inference pipeline:** this repository focuses on notebook-based experimentation, not deployment, distributed training or efficient cached generation.
- **No pretrained weights:** the notebooks train models from random initialization.

## Potential improvements

- Add early stopping and save the best validation-loss checkpoint.
- Persist the tokenizer vocabulary and model configuration.
- Add a clean local training script, `requirements.txt`, and reproducible dataset loading.
- Experiment with learning-rate scheduling, different context sizes and model depths.
- Implement temperature/top-k sampling and key/value caching for generation.

**Author:** [Darius-Cos](https://github.com/Darius-Cos)

---

*An educational project documenting the transition from a simple character-level baseline to a working GPT-style Transformer.*
