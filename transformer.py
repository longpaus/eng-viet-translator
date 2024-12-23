import torch
import torch.nn.functional as F
from torch import nn
from encoder import Encoder
from decoder import Decoder
from embeddings import PositionalEmbedding


class EncoderDecoder(nn.Module):

    def __init__(self, d_model: int, h: int, vocab_size: int, max_seq_len: int, N: int, d_ff: int):
        super(EncoderDecoder, self).__init__()
        self.embedding = PositionalEmbedding(d_model, vocab_size, max_seq_len)
        self.encoder = Encoder(d_model, h, d_ff, N)
        self.decoder = Decoder(d_model, h, d_ff, N)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, memory_mask, src_mask, tgt_mask):
        src_embeddings = self.embedding(src)
        tgt_embeddings = self.embedding(tgt)

        encoder_out = self.encoder(src_embeddings, src_mask)

        decoder_out = self.decoder(tgt_embeddings, encoder_out, memory_mask, tgt_mask)
        logits = self.linear(decoder_out)  # (batch_size,max_seq_len,vocab_size)
        return logits

    def encode(self, src, src_mask):
        return self.encoder(self.embedding(src), src_mask)

    def decode(self, tgt, encoder_output, memory_mask, tgt_mask):
        return self.linear(self.decoder(self.embedding(tgt), encoder_output, memory_mask, tgt_mask))
