from transformer import EncoderDecoder
from utils import create_causal_mask, create_padding_mask
import torch
import sentencepiece as spm
import sys


def greedy_decode(model:EncoderDecoder, src, max_len, pad_token, bos_token, eos_token):
    """
    Decodes a sequence from the model using greedy decoding.

    Args:
        model (EncoderDecoder): The transformer model.
        src (torch.Tensor): The input sequence tensor. Shape: (batch_size, seq_len)
        max_len (int): The maximum length of the decoded sequence.
        pad_token (int): The padding token ID.
        bos_token (int): The beginning-of-sequence token ID.
        eos_token (int): The end-of-sequence token ID.

    Returns:
        torch.Tensor: The decoded sequence tensor. Shape: (batch_size, seq_len)
    """
    src_mask = create_padding_mask(src, pad_token)
    memory = model.encode(src, src_mask)
    device = src.device
    tgt = torch.tensor([[bos_token]], dtype=torch.long, device=src.device)
    
    memory_mask = create_padding_mask(src, pad_token)
    
    for _ in range(max_len - 1):
        seq_len = tgt.shape[1]
        tgt_mask = create_causal_mask(seq_len).to(device) & create_padding_mask(tgt, pad_token).to(device)
        
        logits = model.decode(tgt, memory, memory_mask, tgt_mask)
        
        next_token_probs = logits[:, -1,:]
        next_token = torch.argmax(next_token_probs, dim=-1)

        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == eos_token:
            break
    return tgt


def main():
    if len(sys.argv) > 1:
        input_string = " ".join(sys.argv[1:])
    else:
        print("No string argument provided.")
        exit(1)
    model_path = "model/translator_model.pth"
    sp = spm.SentencePieceProcessor()
    sp.load("model/sentencePiece_model.model")

    d_model = 512
    h = 8
    vocab_size = len(sp)
    max_seq_len = 128
    N = 4
    d_ff = 2048
    
    pad_token = 0
    bos_token = 1
    eos_token = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EncoderDecoder(d_model, h, vocab_size, max_seq_len, N, d_ff).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    src = torch.tensor(sp.encode([input_string], add_bos=True, add_eos=True)).to(device)
    
    with torch.no_grad():
        tgt = greedy_decode(model, src, 128, pad_token, bos_token, eos_token)[0]

    print(sp.decode(tgt.tolist()))


main()
