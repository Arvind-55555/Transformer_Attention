"""
visualize.py
------------
Utilities to plot attention maps from JSON exports created by export_attention_json.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_attention_matrix(attn_matrix, tokens=None, title=None, outpath=None):
    # attn_matrix: (seq_q, seq_k)
    plt.figure(figsize=(6,5))
    plt.imshow(attn_matrix, aspect='auto', cmap='viridis')
    plt.colorbar()
    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
    if title:
        plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        print("Saved attention plot to", outpath)
    else:
        plt.show()

def load_attention_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
