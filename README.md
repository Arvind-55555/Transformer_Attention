# Transformer — Modular Rebuild (Code-by-Code Explanation)

This project contains a modular, educational implementation of the Transformer model with explicit mapping between the mathematical formulas from "Attention Is All You Need" and code.

## Overview

A pedagogical rebuild of the Transformer architecture with clear correspondence between mathematical concepts from the original paper and their Python implementation. Perfect for understanding the inner workings of attention mechanisms and transformer models.

## Mathematical Formulas and Implementation Mapping

| Component | Formula | Implementation |
|-----------|---------|----------------|
| **Scaled Dot-Product Attention** | `Attention(Q,K,V) = softmax((QKᵀ)/√dₖ) V` | `ScaledDotProductAttention` |
| **Multi-Head Attention** | `headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)`<br>`MultiHead = Concat(head₁,...,headₕ)Wᴼ` | `MultiHeadAttention` |
| **Positional Encodings** | Sinusoidal functions | `PositionalEncoding` |
| **Layer Structure** | Add & Norm + FeedForward | `EncoderLayer` / `DecoderLayer` |

## Variable Definitions

| Symbol | Description |
|--------|-------------|
| `Q` | Query matrix |
| `K` | Key matrix |
| `V` | Value matrix |
| `dₖ` | Dimension of key vectors |
| `Wᵢᵠ`, `Wᵢᴷ`, `Wᵢⱽ` | Weight matrices for head *i* |
| `Wᴼ` | Output projection matrix |
| `h` | Number of attention heads |

## Project Structure

```
src/
├── model.py              # Transformer modules (PositionalEncoding, ScaledDotProductAttention, MultiHeadAttention, FeedForward, EncoderLayer, DecoderLayer, SmallTransformer)
├── data.py               # Toy copy task dataset and loader
├── train.py              # Training script for the small transformer on copy task
├── export_attention_json.py  # Export attention tensors to JSON for interactive visualizer
└── visualize.py          # Plotting utilities to visualize attention matrices

notebooks/                # Original notebooks (if uploaded)
docs/assets/
└── Attention.png         # Architecture diagram (from the paper)
```

## Quick Start

### Train on Copy Task
```bash
python -m src.train --epochs 3 --save_dir checkpoints
```

### Export Attention from Checkpoint
```bash
python -m src.export_attention_json --checkpoint checkpoints/ckpt_epoch1.pt --out attention_epoch1.json
```

### Visualize Attention
```python
from src.visualize import load_attention_json, plot_attention_matrix

# Load exported attention data
data = load_attention_json('attention_epoch1.json')

# Plot layer 0, head 0, encoder self-attention
plot_attention_matrix(
    data['attentions']['encoder_self'][0][0], 
    title='Layer 0 Head 0 - Encoder Self-Attention'
)
```

## Features

- 🧮 **Mathematical Transparency**: Direct mapping from paper formulas to code
- 🏗️ **Modular Architecture**: Independent, reusable transformer components
- 📊 **Visualization Tools**: Interactive attention matrix plotting
- 🎯 **Educational Focus**: Designed for learning and experimentation
- 🔄 **Copy Task Demo**: Simple training task to validate implementation

## Requirements

```bash
pip install torch matplotlib numpy
```

## License

MIT - Arvind@2025
