# Transformer for Chess Moves Autoregressive Generation

This repository implements a KV Cache mechanism in autoregressive transformer models and a KV cache compression algorithm that dynamically reduce the size of the cache to decrease its memory footprint.

---

### Implementation Details

To explore these concepts, I implemented a *decoder-only* transformer model using transformer blocks written in Equinox that are built on a `MultiHeadAttention` layer I implemented from scratch using JAX. The final model includes:

- Two **Embedding** layers covering token and positional embeddings
- Two *pre-normalized* **Transformer Blocks**
- A final two-layer **MultiLayer Perceptron** (MLP) with **GELU** activations
- **Dropout** layers applied after transformer blocks and the first layer of the final MLP
- A *"vanilla"* forward pass used during training and a `decode` method that leverages the KV cache mechanism during inference

The model is trained on the [Lichess chess games dataset](https://www.kaggle.com/datasets/aapohermankoskinen/lichess-1-million-chess-games) to perform autoregressive next-move prediction on sequences of chess moves, where each move is treated as a discrete token.

Moreover, the trained model is evaluated with classic performance metrics (i. e., perplexity and top-1, 3, and 5 accuracy) along with an analysis of cache performance measured with per-token latency, Time To First Token (TTFT), and cumulative latency.

Afterwards, I implemented a $L_2$ Norm-based KV cache compression method following the results obtained by Devoto et al., compression is evaluated on correctness and performance trade-offs

---
### Repository Structure
```
├── preprocess.py   # Functions to preprocess the data and to apply tokenization
├── layers.py       # Implementation of the layers of the model
├── model.py        # Architecture of the final model
├── kv_cache.py     # Implementation of the KV Cache and the cache compression method
├── train_evaluate.py # Functions to train and evaluate the model's performance, along with the functions used to evaluate the KV Cache mechanism and the compression method
└── analysis.ipynb    # Final notebook containing the results relative to the model's training and evaluation and the assessment of the behavior of the KV Cache and the compression mechanism
```

---
### References
https://jax-ml.github.io/scaling-book/inference/

https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms

Devoto, A., Zhao, Y., Scardapane, S., & Minervini, P. (2024). A Simple and Effective $L_2$ Norm-Based Strategy for KV Cache Compression. 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV 2024), 18476–18499. https://doi.org/10.18653/v1/2024.emnlp-main.1027
