"""
export_attention_json.py
-------------------------
Loads a saved checkpoint and exports attention tensors (encoder/decoder) to JSON
for the interactive visualizer.
"""
import torch, json, os, argparse
from src.model import SmallTransformer

def export_attention(checkpoint, out_json, sample=None, device='cpu'):
    ck = torch.load(checkpoint, map_location=device)
    args = ck.get('args', {})
    model = SmallTransformer(vocab_size=args.get('vocab_size', 60),
                             d_model=args.get('d_model', 128),
                             num_heads=args.get('num_heads', 4),
                             num_encoder_layers=args.get('num_encoder_layers', 2),
                             num_decoder_layers=args.get('num_decoder_layers', 2),
                             d_ff=args.get('d_ff', 512),
                             max_len=args.get('max_len', 50))
    model.load_state_dict(ck['model_state'])
    model.eval()
    model.to(device)

    # If no sample is provided, synthesize one
    if sample is None:
        import torch as _torch
        src = _torch.randint(2, args.get('vocab_size', 60), (1, 10)).to(device)
        tgt = src.clone().to(device)
    else:
        src, tgt = sample

    with torch.no_grad():
        logits, attns = model(src, tgt)
    # Convert tensors to nested lists (layers -> heads -> seq_q -> seq_k)
    serializable = {}
    for key, lst in attns.items():
        serializable[key] = []
        for layer_attn in lst:
            # layer_attn: tensor (batch, heads, seq_q, seq_k)
            layer_list = layer_attn.squeeze(0).cpu().tolist()
            serializable[key].append(layer_list)
    with open(out_json, 'w') as f:
        json.dump({"tokens": None, "attentions": serializable}, f)
    print("Exported attention JSON to", out_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    export_attention(args.checkpoint, args.out)
