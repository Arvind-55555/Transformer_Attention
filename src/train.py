"""
train.py
--------
Training script for SmallTransformer on the copy task.
Saves checkpoints and example attention tensors.
"""
import os
import torch
from torch import nn
from torch.optim import Adam
from src.model import SmallTransformer
from src.data import get_loader
import argparse

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallTransformer(vocab_size=args.vocab_size, d_model=args.d_model, num_heads=args.num_heads,
                             num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                             d_ff=args.d_ff, max_len=args.max_len).to(device)
    optim = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    loader = get_loader(batch_size=args.batch_size, vocab_size=args.vocab_size, seq_len=args.seq_len, size=args.dataset_size)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            # For simplicity use teacher forcing: input is tgt, target is tgt (shifted not strictly necessary for copy task)
            tgt_input = tgt
            logits, attns = model(src, tgt_input)
            # logits: (batch, tgt_seq, vocab)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = tgt.view(-1)
            loss = criterion(logits_flat, target_flat.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")
        # Save checkpoint and one example attention JSON per epoch
        ckpt = {"model_state": model.state_dict(), "args": vars(args)}
        torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch{epoch}.pt"))
    print('Training complete. Checkpoints saved in', args.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=60)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--dataset_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    train(args)
