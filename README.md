# Transformer — Modular Rebuild (code-by-code explanation)

This project contains a modular, educational implementation of the Transformer model with explicit mapping
between the mathematical formulas from "Attention Is All You Need" and code.

Structure:
- src/model.py: Transformer modules (PositionalEncoding, ScaledDotProductAttention, MultiHeadAttention, FeedForward, EncoderLayer, DecoderLayer, SmallTransformer).
- src/data.py: Toy copy task dataset and loader.
- src/train.py: Training script to train the small transformer on the copy task.
- src/export_attention_json.py: Export attention tensors from a checkpoint into JSON for the interactive visualizer.
- src/visualize.py: Simple plotting utilities to visualize attention matrices.
- notebooks/: original notebook (if uploaded).
- docs/assets/Attention.png: Architecture diagram (from the paper).

How code maps to math (high level):
- Scaled dot-product attention: Attention(Q,K,V) = softmax((QK^T)/sqrt(d_k)) V  -> implemented in ScaledDotProductAttention
- Multi-head: head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V); MultiHead concatenates outputs and projects with W_o -> implemented in MultiHeadAttention
- Positional encodings: sin/cos functions -> implemented in PositionalEncoding
- Layer structure: Add & Norm and FeedForward -> EncoderLayer/DecoderLayer follow the architecture diagram.

Quickstart (train on copy task):
```bash
python -m src.train --epochs 3 --save_dir checkpoints
```
Export attention from checkpoint:
```bash
python -m src.export_attention_json --checkpoint checkpoints/ckpt_epoch1.pt --out attention_epoch1.json
```
Visualize:
```python
from src.visualize import load_attention_json, plot_attention_matrix
data = load_attention_json('attention_epoch1.json')
# plot layer 0, head 0, encoder_self
plot_attention_matrix(data['attentions']['encoder_self'][0][0], title='Layer0 Head0')
```
