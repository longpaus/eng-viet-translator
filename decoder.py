from torch import nn
from utils import clones
from layers import DecoderLayer


class Decoder(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int, N: int):
        """
        Initializes the Decoder module.

        Args:
            d_model (int): The dimensionality of the model.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the feedforward network.
            N (int): The number of decoder layers.
        """
        super(Decoder, self).__init__()
        self.sublayers = clones(DecoderLayer(d_model, h, d_ff), N)

    def forward(self, x, encoder_out, memory_mask, tgt_mask):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): The input tensor (target sequence embeddings). Shape: (batch_size, seq_len, d_model)
            encoder_out (torch.Tensor): The output tensor from the encoder. Shape: (batch_size, seq_len, d_model)
            memory_mask (torch.Tensor): The padding mask for the encoder output. Shape: (batch_size, 1, 1, seq_len)
            tgt_mask (torch.Tensor): The mask for the target sequence, combining padding and causal masks.
                                    Shape: (batch_size, 1, seq_len, seq_len)

        Returns:
            torch.Tensor: The decoded output tensor. Shape: (batch_size, seq_len, d_model)
        """
        for sublayer in self.sublayers:
            x = sublayer(x, encoder_out, memory_mask, tgt_mask)
        return x
